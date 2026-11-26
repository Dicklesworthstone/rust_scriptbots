//! Publish verification reports beside immutable input evidence.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn entry_path(path: &Path) -> io::Result<PathBuf> {
    let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty());
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "expected a file path"))?;
    Ok(parent.unwrap_or(Path::new(".")).canonicalize()?.join(name))
}

fn database_entries(path: &Path, protected: &mut Vec<PathBuf>) {
    protected.push(path.to_owned());
    protected.push(path.with_extension("manifest.json"));
    for suffix in scriptbots_storage::STORAGE_SIDECAR_SUFFIXES
        .into_iter()
        .chain([".scriptbots-writer.lock"])
    {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        protected.push(name.into());
    }
}

fn collision(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "report destination {} conflicts with input evidence or another report",
            path.display()
        ),
    )
}

fn same_entry(left: &Path, right: &Path, left_exists: bool) -> io::Result<bool> {
    if left == right {
        return Ok(true);
    }
    if left_exists {
        // The caller checks exact file identities for existing destinations.
        return Ok(false);
    }
    // An absent SQLite sidecar has no identity to compare. Reserve alternate
    // case spellings in the same directory, including case-folding mounts.
    let folded = |p: &Path| p.file_name().and_then(|s| s.to_str()).map(str::to_lowercase);
    match (folded(left), folded(right)) {
        (Some(a), Some(b)) if a == b => same_file::is_same_file(
            left.parent().expect("absolute file entry"),
            right.parent().expect("absolute file entry"),
        ),
        _ => Ok(false),
    }
}

// Resolve link targets even when the last component is absent. Creating that
// component could otherwise turn a dangling database sidecar into a live one.
fn include_link_targets(protected: &mut Vec<PathBuf>) -> io::Result<()> {
    let mut index = 0;
    while index < protected.len() {
        let path = &protected[index];
        index += 1;
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let target = path.parent().expect("absolute entry").join(fs::read_link(path)?);
                match entry_path(&target) {
                    Ok(entry) if !protected.contains(&entry) => protected.push(entry),
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn regular_entry_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_file() => Ok(true),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("report destination {} must be a regular file or a new path", path.display()),
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

/// Stage every report before replacing any output. A failing verification still
/// publishes its complete verdict; a refused destination leaves prior bytes intact.
pub(super) fn publish(
    outputs: &[(&Path, &[u8])],
    databases: &[PathBuf],
    spec: Option<&Path>,
) -> io::Result<()> {
    if outputs.is_empty() {
        return Ok(());
    }
    let mut protected = Vec::new();
    if let Some(path) = spec {
        protected.push(entry_path(path)?);
        protected.push(path.canonicalize()?);
    }
    for path in databases {
        // A cohort can override the positional database for every member.
        // Reserve its name too, even when that unused file does not yet exist.
        let entry = match entry_path(path) {
            Ok(entry) => entry,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        database_entries(&entry, &mut protected);
        match path.canonicalize() {
            Ok(resolved) if resolved != entry => database_entries(&resolved, &mut protected),
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }

    include_link_targets(&mut protected)?;
    let mut staged = Vec::with_capacity(outputs.len());
    for &(path, bytes) in outputs {
        let output = entry_path(path)?;
        let exists = regular_entry_exists(&output)?;
        for source in &protected {
            if same_entry(&output, source, exists)? {
                return Err(collision(path));
            }
            if exists {
                match fs::metadata(source) {
                    Ok(meta) if meta.is_file() && same_file::is_same_file(&output, source)? => {
                        return Err(collision(path));
                    }
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
            }
        }
        protected.push(output.clone());
        let parent = output.parent().expect("absolute file entry has a parent");
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        file.write_all(bytes)?;
        file.as_file().sync_all()?;
        staged.push((file, output, exists));
    }

    for (file, output, exists) in staged {
        if exists {
            file.persist(&output).map_err(|error| error.error)?;
        } else {
            file.persist_noclobber(&output).map_err(|error| error.error)?;
        }
        #[cfg(unix)]
        fs::File::open(output.parent().expect("absolute file entry has a parent"))?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protects_all_member_inputs_sidecars_and_existing_reports() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.sqlite");
        let second = dir.path().join("second.sqlite");
        let spec = dir.path().join("budget.json");
        for path in [&first, &second, &spec] {
            fs::write(path, b"original evidence").unwrap();
        }
        let inputs = [first, second.clone()];
        let report = dir.path().join("result.json");
        fs::write(&report, b"previous verdict").unwrap();
        let hard_link = dir.path().join("linked-report.json");
        fs::hard_link(&second, &hard_link).unwrap();

        let mut conflicts = Vec::new();
        database_entries(&second, &mut conflicts);
        conflicts.extend([spec.clone(), hard_link, report.clone()]);
        for conflict in conflicts {
            let before = fs::read(&conflict).ok();
            let result = publish(
                &[(&report, b"replacement"), (&conflict, b"bad")],
                &inputs,
                Some(&spec),
            );
            assert!(result.is_err(), "{}", conflict.display());
            assert_eq!(fs::read(&report).unwrap(), b"previous verdict");
            assert_eq!(fs::read(&conflict).ok(), before);
        }
        publish(&[(&report, b"complete verdict")], &inputs, Some(&spec)).unwrap();
        assert_eq!(fs::read(&report).unwrap(), b"complete verdict");
        assert_eq!(fs::read(&second).unwrap(), b"original evidence");
    }

    #[cfg(unix)]
    #[test]
    fn protects_symlinked_inputs_and_dangling_output_links() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("source.db");
        let linked = dir.path().join("alias.db");
        let dangling = dir.path().join("report.json");
        let missing = dir.path().join("missing");
        fs::write(&db, b"untouched").unwrap();
        symlink(&db, &linked).unwrap();
        symlink(&missing, &dangling).unwrap();
        let sidecar_target = dir.path().join("would-be-wal");
        symlink(&sidecar_target, dir.path().join("source.db-shm")).unwrap();
        let inputs = [linked];
        for dest in [
            sidecar_target.clone(),
            db.clone(),
            dir.path().join("source.db-wal"),
            dir.path().join("alias.db-wal"),
            dangling,
        ] {
            assert!(publish(&[(&dest, b"report")], &inputs, None).is_err());
        }
        assert_eq!(fs::read(&db).unwrap(), b"untouched");
        assert!(!missing.exists());
        assert!(!sidecar_target.exists());
    }
}
