//! Brain-family tournament harness, Elo ratings, and leaderboard generator (bd-16g.12).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result record for a match between two or more brain families.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub seed: u64,
    pub ticks: u64,
    pub family_scores: HashMap<String, FamilyScore>,
}

/// Multi-axis performance score for a brain family in a tournament match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyScore {
    pub survival_share: f32,
    pub biomass_share: f32,
    pub max_generation: u32,
}

/// Elo rating record for a brain family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloRating {
    pub family_id: String,
    pub rating: f64,
    pub matches_played: u32,
    pub wins: u32,
}

impl EloRating {
    pub fn new(family_id: impl Into<String>) -> Self {
        Self {
            family_id: family_id.into(),
            rating: 1500.0,
            matches_played: 0,
            wins: 0,
        }
    }

    /// Update Elo ratings for winner vs loser.
    pub fn update_elo(winner: &mut Self, loser: &mut Self, k_factor: f64) {
        let expected_w = 1.0 / (1.0 + 10.0_f64.powf((loser.rating - winner.rating) / 400.0));
        let expected_l = 1.0 / (1.0 + 10.0_f64.powf((winner.rating - loser.rating) / 400.0));

        winner.rating += k_factor * (1.0 - expected_w);
        loser.rating += k_factor * (0.0 - expected_l);

        winner.matches_played += 1;
        loser.matches_played += 1;
        winner.wins += 1;
    }
}

/// Tournament harness running matched-world competitions.
#[derive(Debug, Clone, Default)]
pub struct TournamentHarness {
    pub ratings: HashMap<String, EloRating>,
    pub match_history: Vec<MatchResult>,
}

impl TournamentHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_family(&mut self, family_id: impl Into<String>) {
        let fid = family_id.into();
        self.ratings
            .entry(fid.clone())
            .or_insert_with(|| EloRating::new(fid));
    }

    pub fn record_match(&mut self, result: MatchResult) {
        // Record match and update ratings for top 2 families
        let mut sorted: Vec<_> = result.family_scores.iter().collect();
        sorted.sort_by(|a, b| b.1.survival_share.partial_cmp(&a.1.survival_share).unwrap());

        if sorted.len() >= 2 {
            let winner_id = sorted[0].0;
            let loser_id = sorted[1].0;

            if let (Some(w), Some(l)) = (
                self.ratings.get_mut(winner_id),
                self.ratings.get_mut(loser_id),
            ) {
                let mut w_clone = w.clone();
                let mut l_clone = l.clone();
                EloRating::update_elo(&mut w_clone, &mut l_clone, 32.0);
                *w = w_clone;
                *l = l_clone;
            }
        }
        self.match_history.push(result);
    }

    pub fn generate_leaderboard_markdown(&self) -> String {
        let mut sorted_ratings: Vec<_> = self.ratings.values().collect();
        sorted_ratings.sort_by(|a, b| b.rating.partial_cmp(&a.rating).unwrap());

        let mut out = String::from(
            "# ScriptBots Brain Family Tournament Leaderboard\n\n\
             | Rank | Family ID | Elo Rating | Matches | Win Rate |\n\
             | :--- | :--- | :--- | :--- | :--- |\n",
        );

        for (i, r) in sorted_ratings.iter().enumerate() {
            let win_rate = if r.matches_played > 0 {
                (r.wins as f64 / r.matches_played as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "| {} | {} | {:.1} | {} | {:.1}% |\n",
                i + 1,
                r.family_id,
                r.rating,
                r.matches_played,
                win_rate
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elo_update_math() {
        let mut mlp = EloRating::new("mlp");
        let mut dwraon = EloRating::new("dwraon");

        EloRating::update_elo(&mut mlp, &mut dwraon, 32.0);

        assert!(mlp.rating > 1500.0);
        assert!(dwraon.rating < 1500.0);
        assert_eq!(mlp.wins, 1);
    }

    #[test]
    fn test_tournament_harness_leaderboard_generation() {
        let mut harness = TournamentHarness::new();
        harness.register_family("mlp");
        harness.register_family("dwraon");

        let mut scores = HashMap::new();
        scores.insert(
            "mlp".to_string(),
            FamilyScore {
                survival_share: 0.7,
                biomass_share: 0.65,
                max_generation: 15,
            },
        );
        scores.insert(
            "dwraon".to_string(),
            FamilyScore {
                survival_share: 0.3,
                biomass_share: 0.35,
                max_generation: 12,
            },
        );

        harness.record_match(MatchResult {
            seed: 42,
            ticks: 5000,
            family_scores: scores,
        });

        let leaderboard = harness.generate_leaderboard_markdown();
        assert!(leaderboard.contains("mlp"));
        assert!(leaderboard.contains("dwraon"));
        assert!(leaderboard.contains("Elo Rating"));
    }
}
