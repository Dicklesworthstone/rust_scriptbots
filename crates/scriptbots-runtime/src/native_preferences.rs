//! Per-user preferences shared by the native terminal and graphical interfaces.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

/// Directory used for ScriptBots UI preferences, independent of the working directory.
#[must_use]
pub fn config_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")));
    base.filter(|path| path.is_absolute())
        .map(|path| path.join("scriptbots"))
}

/// Claim the one automatic guide opening shared by native interactive frontends.
/// A returning user or a profile without writable preferences can open it manually.
#[must_use]
pub fn claim_first_lab_guide_launch() -> bool {
    config_dir().is_some_and(|dir| claim_in(&dir).unwrap_or(false))
}

fn claim_in(dir: &Path) -> io::Result<bool> {
    fs::create_dir_all(dir)?;
    // The marker's existence is the entire preference. Exclusive creation
    // prevents two concurrently launched frontends from both claiming it.
    match fs::File::create_new(dir.join("lab-guide-seen-v1")) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_frontends_offer_the_guide_once_per_profile() {
        let mut salt = 0_u64;
        let root = loop {
            let path = std::env::temp_dir()
                .join(format!("scriptbots-guide-{}-{salt}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => salt += 1,
                Err(error) => panic!("create temporary profile: {error}"),
            }
        };
        let dir = root.join("profile");
        let barrier = std::sync::Barrier::new(4);
        let claimed = std::thread::scope(|scope| {
            let attempts: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        claim_in(&dir).unwrap()
                    })
                })
                .collect();
            attempts
                .into_iter()
                .map(|attempt| attempt.join().unwrap())
                .filter(|first| *first)
                .count()
        });
        assert_eq!(claimed, 1);
        assert!(!claim_in(&dir).unwrap());
        assert!(claim_in(&root.join("second-profile")).unwrap());
        assert!(claim_in(&dir.join("lab-guide-seen-v1")).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
