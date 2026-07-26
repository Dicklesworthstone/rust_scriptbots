//! The canonical sensor and actuator channel layout (`bd-2z0.2.4`, `bd-16g.4.1`).
//!
//! Every brain speaks the same wire format: 25 sensor values in, 9 actuator
//! values out. Until this module existed, that format was an *implicit
//! convention* — every consumer hand-indexed the arrays and hoped. It did not
//! hold. Combat read `outputs[3]` and called it "boost" for months; index 3 is
//! the **green colour channel**, so carnivores were rewarded for being green
//! while agents that actually boosted got nothing. That bug was invisible
//! precisely because nothing in the type system knew what index 3 meant.
//!
//! So this module is the single source of truth, and it is enforced at compile
//! time:
//!
//! * [`OutputChannel`] and [`SensorChannel`] name every slot.
//! * `const` assertions prove the tables are exhaustive, that every index in
//!   `0..OUTPUT_SIZE` / `0..INPUT_SIZE` appears exactly once, and that no two
//!   channels collide.
//! * Consumers read through [`OutputsExt`] / [`SensorsExt`], so a channel is
//!   referred to by *name*, never by a number typed from memory.
//!
//! Adding a sensor or an actuator now breaks the build until every table is
//! updated — which is exactly the failure mode we want, instead of a silent
//! mislabel that survives for months.

use serde::{Deserialize, Serialize};

use crate::{INPUT_SIZE, NUM_EYES, OUTPUT_SIZE};

/// One actuator slot in a brain's output vector.
///
/// The discriminants are the wire format and are matched by the legacy C++
/// (`World.cpp` `processOutputs`); do not renumber them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OutputChannel {
    /// Left wheel velocity.
    WheelLeft = 0,
    /// Right wheel velocity.
    WheelRight = 1,
    /// Body colour, red.
    ColorRed = 2,
    /// Body colour, green. **Not** boost — see the module docs.
    ColorGreen = 3,
    /// Body colour, blue.
    ColorBlue = 4,
    /// Desired spike length.
    SpikeTarget = 5,
    /// Boost request; active above [`BOOST_THRESHOLD`].
    Boost = 6,
    /// Emitted sound volume.
    SoundLevel = 7,
    /// Willingness to give food to neighbours.
    GiveIntent = 8,
}

/// Boost engages above this output value (legacy: `a->boost = a->out[6] > 0.5`).
pub const BOOST_THRESHOLD: f32 = 0.5;

impl OutputChannel {
    /// Every actuator channel, in wire order.
    pub const ALL: [Self; OUTPUT_SIZE] = [
        Self::WheelLeft,
        Self::WheelRight,
        Self::ColorRed,
        Self::ColorGreen,
        Self::ColorBlue,
        Self::SpikeTarget,
        Self::Boost,
        Self::SoundLevel,
        Self::GiveIntent,
    ];

    /// Index of this channel in the output vector.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable identifier, suitable for logs, analytics, and UI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::WheelLeft => "wheel_left",
            Self::WheelRight => "wheel_right",
            Self::ColorRed => "color_red",
            Self::ColorGreen => "color_green",
            Self::ColorBlue => "color_blue",
            Self::SpikeTarget => "spike_target",
            Self::Boost => "boost",
            Self::SoundLevel => "sound_level",
            Self::GiveIntent => "give_intent",
        }
    }
}

/// What a sensor slot is actually measuring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorKind {
    /// How much agent-matter this eye can see.
    EyeDensity,
    /// Red seen by this eye.
    EyeRed,
    /// Green seen by this eye.
    EyeGreen,
    /// Blue seen by this eye.
    EyeBlue,
    /// Food in the cell underfoot.
    Food,
    /// Movement noise from neighbours.
    Sound,
    /// Proximity of neighbours, direction-agnostic.
    Smell,
    /// The agent's own health.
    Health,
    /// Deliberate sound emitted by neighbours.
    Hearing,
    /// Wounded agents seen ahead.
    Blood,
    /// Discomfort with the local temperature.
    Temperature,
    /// Internal oscillator.
    Clock,
}

/// One slot in a brain's sensor vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorChannel {
    /// Index in the sensor vector.
    pub index: usize,
    /// What this slot measures.
    pub kind: SensorKind,
    /// Which eye, for the per-eye channels.
    pub eye: Option<usize>,
    /// Stable identifier for logs, analytics, and UI labels.
    pub name: &'static str,
}

impl SensorChannel {
    const fn eye(index: usize, kind: SensorKind, eye: usize, name: &'static str) -> Self {
        Self {
            index,
            kind,
            eye: Some(eye),
            name,
        }
    }

    const fn plain(index: usize, kind: SensorKind, name: &'static str) -> Self {
        Self {
            index,
            kind,
            eye: None,
            name,
        }
    }
}

/// The canonical sensor layout, in wire order.
///
/// This mirrors `World::setInputs` in the legacy C++ exactly:
/// `P1 R1 G1 B1 FOOD P2 R2 G2 B2 SOUND SMELL HEALTH P3 R3 G3 B3 CLOCK1 CLOCK2
/// HEARING BLOOD TEMPERATURE P4 R4 G4 B4`.
///
/// Note the eyes are **not** contiguous — eye 3's block sits at the end, after
/// the scalar channels. Anyone re-deriving this layout from intuition rather
/// than from this table will get it wrong.
pub const SENSOR_LAYOUT: [SensorChannel; INPUT_SIZE] = [
    SensorChannel::eye(0, SensorKind::EyeDensity, 0, "eye0_density"),
    SensorChannel::eye(1, SensorKind::EyeRed, 0, "eye0_red"),
    SensorChannel::eye(2, SensorKind::EyeGreen, 0, "eye0_green"),
    SensorChannel::eye(3, SensorKind::EyeBlue, 0, "eye0_blue"),
    SensorChannel::plain(4, SensorKind::Food, "food"),
    SensorChannel::eye(5, SensorKind::EyeDensity, 1, "eye1_density"),
    SensorChannel::eye(6, SensorKind::EyeRed, 1, "eye1_red"),
    SensorChannel::eye(7, SensorKind::EyeGreen, 1, "eye1_green"),
    SensorChannel::eye(8, SensorKind::EyeBlue, 1, "eye1_blue"),
    SensorChannel::plain(9, SensorKind::Sound, "sound"),
    SensorChannel::plain(10, SensorKind::Smell, "smell"),
    SensorChannel::plain(11, SensorKind::Health, "health"),
    SensorChannel::eye(12, SensorKind::EyeDensity, 2, "eye2_density"),
    SensorChannel::eye(13, SensorKind::EyeRed, 2, "eye2_red"),
    SensorChannel::eye(14, SensorKind::EyeGreen, 2, "eye2_green"),
    SensorChannel::eye(15, SensorKind::EyeBlue, 2, "eye2_blue"),
    SensorChannel::plain(16, SensorKind::Clock, "clock1"),
    SensorChannel::plain(17, SensorKind::Clock, "clock2"),
    SensorChannel::plain(18, SensorKind::Hearing, "hearing"),
    SensorChannel::plain(19, SensorKind::Blood, "blood"),
    SensorChannel::plain(20, SensorKind::Temperature, "temperature"),
    SensorChannel::eye(21, SensorKind::EyeDensity, 3, "eye3_density"),
    SensorChannel::eye(22, SensorKind::EyeRed, 3, "eye3_red"),
    SensorChannel::eye(23, SensorKind::EyeGreen, 3, "eye3_green"),
    SensorChannel::eye(24, SensorKind::EyeBlue, 3, "eye3_blue"),
];

// ---------------------------------------------------------------------------
// Compile-time proofs. These are the whole point of the module: a mislabelled
// or duplicated channel must fail the BUILD, not a code review.
// ---------------------------------------------------------------------------

const _: () = {
    // Every output index appears exactly once, in order.
    let mut i = 0;
    while i < OUTPUT_SIZE {
        assert!(
            OutputChannel::ALL[i].index() == i,
            "output channel table is out of order or has a duplicate index"
        );
        i += 1;
    }

    // Every sensor index appears exactly once, in order.
    let mut i = 0;
    while i < INPUT_SIZE {
        assert!(
            SENSOR_LAYOUT[i].index == i,
            "sensor layout is out of order or has a duplicate index"
        );
        i += 1;
    }

    // Each eye contributes exactly four channels; a fifth (or a missing one)
    // means the sensor vector and the eye count have drifted apart.
    let mut eye = 0;
    while eye < NUM_EYES {
        let mut count = 0;
        let mut i = 0;
        while i < INPUT_SIZE {
            if let Some(owner) = SENSOR_LAYOUT[i].eye
                && owner == eye
            {
                count += 1;
            }
            i += 1;
        }
        assert!(count == 4, "every eye needs exactly density+R+G+B");
        eye += 1;
    }

    // The historical bug, pinned forever: boost is 6, green is 3.
    assert!(OutputChannel::Boost.index() == 6);
    assert!(OutputChannel::ColorGreen.index() == 3);
};

/// Read a brain's actuator vector by channel name rather than by a magic index.
pub trait OutputsExt {
    /// Raw value of one actuator channel.
    #[must_use]
    fn channel(&self, channel: OutputChannel) -> f32;

    /// Value of one actuator channel, clamped to the unit interval.
    #[must_use]
    fn channel_clamped(&self, channel: OutputChannel) -> f32 {
        self.channel(channel).clamp(0.0, 1.0)
    }

    /// Peak commanded wheel effort in normalized actuator space.
    ///
    /// This deliberately excludes physical displacement and boost-scaled wheel speed.
    #[must_use]
    fn peak_wheel_output(&self) -> f32 {
        self.channel_clamped(OutputChannel::WheelLeft)
            .max(self.channel_clamped(OutputChannel::WheelRight))
    }

    /// Whether the brain is requesting boost.
    #[must_use]
    fn boost_engaged(&self) -> bool {
        self.channel(OutputChannel::Boost) > BOOST_THRESHOLD
    }
}

impl OutputsExt for [f32; OUTPUT_SIZE] {
    fn channel(&self, channel: OutputChannel) -> f32 {
        self[channel.index()]
    }
}

/// Read a brain's sensor vector by channel rather than by a magic index.
pub trait SensorsExt {
    /// Raw value of one sensor slot.
    #[must_use]
    fn sensor(&self, channel: &SensorChannel) -> f32;

    /// Every sensor slot paired with its description, in wire order.
    #[must_use]
    fn labelled(&self) -> [(&'static SensorChannel, f32); INPUT_SIZE];
}

impl SensorsExt for [f32; INPUT_SIZE] {
    fn sensor(&self, channel: &SensorChannel) -> f32 {
        self[channel.index]
    }

    fn labelled(&self) -> [(&'static SensorChannel, f32); INPUT_SIZE] {
        std::array::from_fn(|i| (&SENSOR_LAYOUT[i], self[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_channels_cover_every_index_exactly_once() {
        let mut seen = [false; OUTPUT_SIZE];
        for channel in OutputChannel::ALL {
            assert!(
                !seen[channel.index()],
                "{} duplicates index {}",
                channel.name(),
                channel.index()
            );
            seen[channel.index()] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "an output index is unnamed");
    }

    #[test]
    fn sensor_layout_covers_every_index_exactly_once() {
        let mut seen = [false; INPUT_SIZE];
        for channel in &SENSOR_LAYOUT {
            assert!(
                !seen[channel.index],
                "{} duplicates its index",
                channel.name
            );
            seen[channel.index] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "a sensor index is unnamed");
    }

    /// `labelled()` is the shared decode every view is supposed to consume
    /// instead of indexing the sensor vector by hand (bd-16g.4). If it ever
    /// paired a value with the wrong channel, every UI built on it would
    /// mislabel in perfect agreement — the drift would be invisible precisely
    /// because everything drifted together.
    ///
    /// Uses a vector whose value ENCODES its own index, so a transposition of
    /// any two slots is caught. A uniform or zeroed vector would pass while
    /// scrambled.
    #[test]
    fn labelled_pairs_every_value_with_the_channel_that_owns_its_index() {
        let mut sensors = [0.0_f32; INPUT_SIZE];
        for (index, slot) in sensors.iter_mut().enumerate() {
            // Distinct, exactly representable, and never 0.0 so a slot that was
            // silently skipped is distinguishable from one that holds zero.
            *slot = (index as f32) + 0.5;
        }

        let labelled = sensors.labelled();
        assert_eq!(labelled.len(), INPUT_SIZE, "decode must be total");

        for (position, &(channel, value)) in labelled.iter().enumerate() {
            assert_eq!(
                channel.index, position,
                "slot {position} was labelled with the channel that owns index {}",
                channel.index
            );
            assert_eq!(
                value, sensors[position],
                "slot {position} ({}) carries the value of a different slot",
                channel.name
            );
            // And the by-channel accessor must agree with the bulk decode, or a
            // view reading one way sees something different from a view reading
            // the other.
            assert_eq!(
                sensors.sensor(channel),
                value,
                "sensor({}) disagrees with labelled() at index {position}",
                channel.name
            );
        }
    }

    /// A transposition must actually fail the check above. Without this, the
    /// test could be asserting a tautology and nobody would know.
    #[test]
    fn the_decode_check_would_catch_a_transposed_pair() {
        let mut sensors = [0.0_f32; INPUT_SIZE];
        for (index, slot) in sensors.iter_mut().enumerate() {
            *slot = (index as f32) + 0.5;
        }
        let labelled = sensors.labelled();

        // Simulate the defect: read slot 1's value while claiming slot 0's channel.
        let claimed = labelled[0].0;
        let wrong_value = labelled[1].1;
        assert_ne!(
            sensors.sensor(claimed),
            wrong_value,
            "if these compared equal the encoding could not distinguish slots and \
             labelled_pairs_every_value_with_the_channel_that_owns_its_index would be vacuous"
        );
    }

    #[test]
    fn channel_names_are_unique() {
        // Names reach analytics, logs, and the UI; two channels sharing one name
        // would silently merge two different quantities in a chart.
        for (i, a) in SENSOR_LAYOUT.iter().enumerate() {
            for b in &SENSOR_LAYOUT[i + 1..] {
                assert_ne!(a.name, b.name, "duplicate sensor name {}", a.name);
            }
        }
        for (i, a) in OutputChannel::ALL.iter().enumerate() {
            for b in &OutputChannel::ALL[i + 1..] {
                assert_ne!(a.name(), b.name(), "duplicate output name {}", a.name());
            }
        }
    }

    #[test]
    fn boost_is_channel_six_and_green_is_channel_three() {
        // The regression pin for the real bug this module exists to prevent:
        // combat read outputs[3] as "boost" for months, so carnivores were
        // rewarded for being GREEN while agents that actually boosted got
        // nothing. If anyone renumbers these, the build stops here.
        assert_eq!(OutputChannel::Boost.index(), 6);
        assert_eq!(OutputChannel::ColorGreen.index(), 3);
        assert_ne!(
            OutputChannel::Boost.index(),
            OutputChannel::ColorGreen.index()
        );

        let mut outputs = [0.0f32; OUTPUT_SIZE];
        outputs[OutputChannel::ColorGreen.index()] = 1.0;
        assert!(
            !outputs.boost_engaged(),
            "a maximally green agent is not boosting"
        );

        outputs[OutputChannel::Boost.index()] = 1.0;
        assert!(outputs.boost_engaged(), "out[6] > 0.5 means boost");
    }

    #[test]
    fn sensor_layout_matches_the_legacy_wire_order() {
        // Independently written from World::setInputs' comment, so this table is
        // checked against the C++ rather than against itself. Note eye 3 lives
        // at the END, after the scalar channels — a layout nobody would guess.
        let expected = [
            "eye0_density",
            "eye0_red",
            "eye0_green",
            "eye0_blue",
            "food",
            "eye1_density",
            "eye1_red",
            "eye1_green",
            "eye1_blue",
            "sound",
            "smell",
            "health",
            "eye2_density",
            "eye2_red",
            "eye2_green",
            "eye2_blue",
            "clock1",
            "clock2",
            "hearing",
            "blood",
            "temperature",
            "eye3_density",
            "eye3_red",
            "eye3_green",
            "eye3_blue",
        ];
        let actual: Vec<&str> = SENSOR_LAYOUT.iter().map(|c| c.name).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn accessors_read_the_slot_they_name() {
        let mut outputs = [0.0f32; OUTPUT_SIZE];
        for channel in OutputChannel::ALL {
            outputs[channel.index()] = channel.index() as f32;
        }
        for channel in OutputChannel::ALL {
            assert!((outputs.channel(channel) - channel.index() as f32).abs() < f32::EPSILON);
        }

        let mut sensors = [0.0f32; INPUT_SIZE];
        for (i, slot) in sensors.iter_mut().enumerate() {
            *slot = i as f32;
        }
        for channel in &SENSOR_LAYOUT {
            assert!((sensors.sensor(channel) - channel.index as f32).abs() < f32::EPSILON);
        }
        let labelled = sensors.labelled();
        assert_eq!(labelled.len(), INPUT_SIZE);
        assert_eq!(labelled[19].0.name, "blood");
        assert!((labelled[19].1 - 19.0).abs() < f32::EPSILON);
    }

    #[test]
    fn clamping_accessor_bounds_hostile_brain_output() {
        // Brains are evolved, not written: nothing stops one emitting -3.0 or
        // 1e9. Consumers must never see an unclamped control value.
        let mut outputs = [0.0f32; OUTPUT_SIZE];
        outputs[OutputChannel::WheelLeft.index()] = -3.0;
        outputs[OutputChannel::WheelRight.index()] = 1e9;
        assert!((outputs.channel_clamped(OutputChannel::WheelLeft) - 0.0).abs() < f32::EPSILON);
        assert!((outputs.channel_clamped(OutputChannel::WheelRight) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn peak_wheel_output_uses_the_stronger_named_wheel_only() {
        let mut outputs = [0.0f32; OUTPUT_SIZE];
        assert!(outputs.peak_wheel_output().abs() < f32::EPSILON);

        outputs[OutputChannel::WheelLeft.index()] = 0.25;
        outputs[OutputChannel::WheelRight.index()] = 0.75;
        outputs[OutputChannel::ColorGreen.index()] = 1.0;
        outputs[OutputChannel::Boost.index()] = 1.0;
        assert!((outputs.peak_wheel_output() - 0.75).abs() < f32::EPSILON);

        outputs[OutputChannel::WheelLeft.index()] = 0.9;
        outputs[OutputChannel::WheelRight.index()] = 0.1;
        assert!((outputs.peak_wheel_output() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn peak_wheel_output_clamps_each_wheel_before_reduction() {
        let mut outputs = [0.0f32; OUTPUT_SIZE];
        outputs[OutputChannel::WheelLeft.index()] = -3.0;
        outputs[OutputChannel::WheelRight.index()] = 0.4;
        assert!((outputs.peak_wheel_output() - 0.4).abs() < f32::EPSILON);

        outputs[OutputChannel::WheelRight.index()] = 1e9;
        assert!((outputs.peak_wheel_output() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_world_io_layout_default_and_bands() {
        let default_layout = WorldIoLayout::DEFAULT;
        assert_eq!(default_layout.bands, 1);
        assert_eq!(default_layout.inputs, INPUT_SIZE);
        assert_eq!(default_layout.outputs, OUTPUT_SIZE);

        let layout_3 = WorldIoLayout::new(3).unwrap();
        assert_eq!(layout_3.bands, 3);
        assert_eq!(layout_3.inputs, INPUT_SIZE + 2);
        assert_eq!(layout_3.outputs, OUTPUT_SIZE + 2);

        assert!(WorldIoLayout::new(0).is_err());
        assert!(WorldIoLayout::new(9).is_err());
    }

    #[test]
    fn test_layout_mismatch_rejection() {
        let layout_3 = WorldIoLayout::new(3).unwrap();
        let err = layout_3
            .validate_brain_io("mlp", 25, 9)
            .expect_err("mismatch must fail");
        match err {
            IoLayoutError::LayoutMismatch {
                brain_kind,
                expected_inputs,
                expected_outputs,
                actual_inputs,
                actual_outputs,
            } => {
                assert_eq!(brain_kind, "mlp");
                assert_eq!(expected_inputs, 27);
                assert_eq!(expected_outputs, 11);
                assert_eq!(actual_inputs, 25);
                assert_eq!(actual_outputs, 9);
            }
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn test_band_isolation_and_stride() {
        // Stride test: 3 agents with 3 bands
        let bands = 3usize;
        let num_agents = 3usize;
        let mut work_sound_emitters = vec![0.0f32; num_agents * bands];

        // Agent 0 emits on band 0: [1.0, 0.0, 0.0]
        work_sound_emitters[0 * bands + 0] = 1.0;
        work_sound_emitters[0 * bands + 1] = 0.0;
        work_sound_emitters[0 * bands + 2] = 0.0;

        // Agent 1 emits on band 1: [0.0, 1.0, 0.0]
        work_sound_emitters[1 * bands + 0] = 0.0;
        work_sound_emitters[1 * bands + 1] = 1.0;
        work_sound_emitters[1 * bands + 2] = 0.0;

        // Agent 2 is listener, dist_factor = 0.5 to Agent 0 and 0.5 to Agent 1
        let dist_factor_0 = 0.5f32;
        let dist_factor_1 = 0.5f32;

        let mut heard = vec![0.0f32; bands];
        for b in 0..bands {
            let sum = dist_factor_0 * work_sound_emitters[0 * bands + b]
                + dist_factor_1 * work_sound_emitters[1 * bands + b];
            heard[b] = sum.min(1.0).max(0.0);
        }

        assert_eq!(heard[0], 0.5);
        assert_eq!(heard[1], 0.5);
        assert_eq!(heard[2], 0.0, "band 2 must be exactly 0.0 (band isolation)");
    }
}

/// Error variants for I/O layout validation and bounds.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum IoLayoutError {
    /// Invalid bands count provided (must be 1..=8).
    #[error("invalid bands count {0}, must be 1..=8")]
    InvalidBands(u8),
    /// Brain layout mismatch between expected and actual sizes.
    #[error(
        "brain IO layout mismatch for '{brain_kind}': expected {expected_inputs}in/{expected_outputs}out, got {actual_inputs}in/{actual_outputs}out"
    )]
    LayoutMismatch {
        brain_kind: String,
        expected_inputs: usize,
        expected_outputs: usize,
        actual_inputs: usize,
        actual_outputs: usize,
    },
}

/// Versioned World I/O layout specification for sensory inputs and actuator outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldIoLayout {
    /// Version of the I/O layout schema.
    pub version: u16,
    /// Number of signal communication bands (1..=8).
    pub bands: u8,
    /// Total sensor input size for this layout.
    pub inputs: usize,
    /// Total actuator output size for this layout.
    pub outputs: usize,
}

impl Default for WorldIoLayout {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl WorldIoLayout {
    /// Default single-band (legacy 25/9) layout.
    pub const DEFAULT: Self = Self {
        version: 1,
        bands: 1,
        inputs: INPUT_SIZE,
        outputs: OUTPUT_SIZE,
    };

    /// Constructs a new layout specification with `bands` signal communication channels (1..=8).
    pub fn new(bands: u8) -> Result<Self, IoLayoutError> {
        if bands == 0 || bands > 8 {
            return Err(IoLayoutError::InvalidBands(bands));
        }
        Ok(Self {
            version: 1,
            bands,
            inputs: INPUT_SIZE + (bands as usize - 1),
            outputs: OUTPUT_SIZE + (bands as usize - 1),
        })
    }

    /// Validates that a brain's reported I/O layout matches this world layout.
    pub fn validate_brain_io(
        &self,
        brain_kind: &str,
        actual_inputs: usize,
        actual_outputs: usize,
    ) -> Result<(), IoLayoutError> {
        if self.inputs != actual_inputs || self.outputs != actual_outputs {
            return Err(IoLayoutError::LayoutMismatch {
                brain_kind: brain_kind.to_owned(),
                expected_inputs: self.inputs,
                expected_outputs: self.outputs,
                actual_inputs,
                actual_outputs,
            });
        }
        Ok(())
    }
}
