//! Manual inspector focus follows a run-stable creature UID across snapshots.

#[derive(Debug, Default)]
pub(super) enum ManualFocus {
    #[default]
    Initial,
    Agent {
        row: usize,
        uid: u64,
    },
    Cleared {
        row: usize,
    },
}

impl ManualFocus {
    fn row(&self) -> usize {
        match *self {
            Self::Initial => 0,
            Self::Agent { row, .. } | Self::Cleared { row } => row,
        }
    }

    pub(super) fn clear(&mut self) {
        *self = Self::Cleared { row: self.row() };
    }

    /// Choose from the displayed population before a refresh can reorder it.
    pub(super) fn browse<T>(
        &mut self,
        agents: &[T],
        displayed: Option<u64>,
        offset: isize,
        uid_of: impl Fn(&T) -> Option<u64>,
    ) {
        if agents.is_empty() {
            return;
        }
        let origin = displayed
            .and_then(|uid| agents.iter().position(|agent| uid_of(agent) == Some(uid)))
            .unwrap_or_else(|| self.row().min(agents.len() - 1));
        let row = origin.saturating_add_signed(offset) % agents.len();
        *self = match uid_of(&agents[row]) {
            Some(uid) => Self::Agent { row, uid },
            None => Self::Cleared { row },
        };
    }

    pub(super) fn resolve<'a, T>(
        &mut self,
        agents: &'a [T],
        uid_of: impl Fn(&T) -> u64,
    ) -> Option<&'a T> {
        match *self {
            Self::Initial => {
                let first = agents.first()?;
                *self = Self::Agent {
                    row: 0,
                    uid: uid_of(first),
                };
                Some(first)
            }
            Self::Agent { row, uid } => {
                if let Some(agent) = agents.get(row)
                    && uid_of(agent) == uid
                {
                    return Some(agent);
                }
                let found = agents.iter().enumerate().find(|(_, a)| uid_of(a) == uid);
                if let Some((row, agent)) = found {
                    *self = Self::Agent { row, uid };
                    Some(agent)
                } else {
                    self.clear();
                    None
                }
            }
            Self::Cleared { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ManualFocus;

    #[test]
    fn compaction_reordering_and_slot_reuse_cannot_retarget_manual_focus() {
        let mut focus = ManualFocus::Agent { row: 2, uid: 30 };
        let uid = |v: &u64| *v;
        assert_eq!(focus.resolve(&[10, 20, 30, 40], uid), Some(&30));
        assert_eq!(focus.resolve(&[40, 30, 9], uid), Some(&30));
        assert_eq!(focus.row(), 1);
        assert_eq!(focus.resolve(&[40, 50, 9], uid), None);
        assert_eq!(focus.resolve(&[60, 70], uid), None);
    }

    #[test]
    fn local_navigation_binds_the_displayed_uid_before_the_next_snapshot() {
        let mut focus = ManualFocus::default();
        let uid = |v: &u64| *v;
        assert_eq!(focus.resolve(&[], uid), None);
        assert_eq!(focus.resolve(&[18, 4, 25], uid), Some(&18));
        focus.browse(&[18, 4, 25], Some(18), 1, |v| Some(*v));
        assert_eq!(focus.resolve(&[4, 25], uid), Some(&4));
        focus.clear();
        assert_eq!(focus.resolve(&[4, 25], uid), None);
        focus.browse(&[4, 25], None, 1, |v| Some(*v));
        assert_eq!(focus.resolve(&[25, 4], uid), Some(&25));
    }
}
