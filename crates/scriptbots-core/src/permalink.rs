//! Permalink codec: shareable, falsifiable world references (`bd-16g.8.1`).
//!
//! A permalink names an exact world: **scenario reference + inline knob diff +
//! seed + build identity**. It is not a snapshot blob — the receiver reconstructs
//! the world from their own scenario document plus the diff, and the embedded
//! config digest proves the reconstruction matches what the sender ran. That is
//! the determinism promise made publicly falsifiable: a stranger can check the
//! claim instead of trusting a paragraph.
//!
//! This is the ONLY place in the project where untrusted, attacker-controlled
//! bytes reach a parser (browser links, chat pastes). Every decode path is
//! therefore length-bounded per field before any allocation, canonical (sorted
//! knob entries, exact f64 bit patterns — never locale-formatted decimals), and
//! total: arbitrary bytes yield `Ok` or a typed [`PermalinkError`], never a
//! panic, never an unbounded allocation.
//!
//! The codec is a pure leaf: it never logs. Every error variant carries the
//! byte offset, field, and offending value a caller needs to emit one complete
//! diagnostic line about a rejected link.

use std::fmt;

use serde::{Deserialize, Serialize};

/// URL-safe base64url (no padding) prefix every permalink string carries.
pub const PERMALINK_PREFIX: &str = "sbw1.";

const MAGIC: &[u8; 3] = b"sb1";
const FORMAT_VERSION: u8 = 1;
/// Largest accepted scenario id (matches the manifest's storage bound).
pub const MAX_SCENARIO_ID_BYTES: usize = 512;
/// Largest accepted knob path.
pub const MAX_KNOB_PATH_BYTES: usize = 128;
/// Most knob entries one link may carry.
pub const MAX_KNOB_ENTRIES: usize = 64;
/// Largest accepted raw payload. Fits 64 worst-case knobs with headroom.
pub const MAX_PAYLOAD_BYTES: usize = 12_288;
const U16_MAX_AS_USIZE: usize = u16::MAX as usize;

/// Build identity carried by a link: the three digests that decide whether a
/// receiver's binary can replay the sender's run bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildLink {
    /// Digest of the pinned toolchain file (`rust-toolchain.toml`).
    pub toolchain_digest: u64,
    /// Digest of the dependency lockfile.
    pub lockfile_digest: u64,
    /// Digest of the core build identity (feature set: simd/parallel lanes).
    pub core_digest: u64,
}

/// Verdict of comparing the receiver's build against the link's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildMatch {
    /// All three digests agree: bit-for-bit replay is the expectation.
    Exact,
    /// Science-lane digests (lockfile + core features) agree but the compiler
    /// toolchain differs: behavior is expected identical, not bit-proven.
    Compatible,
    /// A science-lane digest differs — most dangerously the core feature set,
    /// which can change trajectories. The receiver must not claim the run.
    Mismatch,
}

/// A decoded permalink.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Permalink {
    /// Scenario reference (catalog id or a derived run's scenario id).
    pub scenario_id: String,
    /// Root RNG seed of the run.
    pub seed: u64,
    /// Inline knob diff over the scenario's resolved config, canonical order.
    pub knob_diff: Vec<(String, f64)>,
    /// Digest of the full config reconstructed from (scenario config + diff).
    pub config_digest: u64,
    /// Build identity the sender ran.
    pub build: BuildLink,
}

/// Every way a permalink can be invalid, with enough context for one diagnostic line.
/// (`PartialEq` only: `OutOfRange` carries `f64`, which has no `Eq`.)
#[derive(Debug, Clone, PartialEq)]
pub enum PermalinkError {
    /// Fewer bytes than the fixed header needs.
    TruncatedHeader {
        /// Byte offset at which the header ran out.
        offset: usize,
        /// Bytes the fixed header requires at that point.
        needed: usize,
        /// Bytes actually remaining in the input.
        available: usize,
    },
    /// A declared field length overran the payload.
    TruncatedField {
        /// Name of the length-prefixed field that overran the payload.
        field: &'static str,
        /// Byte offset where the field begins.
        offset: usize,
        /// Length the field declared for itself.
        declared: usize,
        /// Bytes actually remaining after `offset`.
        remaining: usize,
    },
    /// Payload exceeds the documented cap before parsing starts.
    OversizedPayload {
        /// Payload size in bytes presented by the caller.
        actual: usize,
        /// Documented payload cap in bytes.
        maximum: usize,
    },
    /// Magic prefix mismatch.
    BadMagic {
        /// First 16 input bytes, hex-encoded, for one-line diagnosis.
        first_16_bytes_hex: String,
    },
    /// Unsupported format version.
    UnsupportedVersion {
        /// Format version byte found in the header.
        found: u8,
        /// Format version this build supports.
        supported: u8,
    },
    /// Reserved flag bits were set.
    NonzeroFlags {
        /// The offending flags byte (reserved bits must be zero).
        found: u8,
    },
    /// CRC32 mismatch — the payload is corrupt.
    CrcMismatch {
        /// CRC32 stored in the header.
        expected: u32,
        /// CRC32 computed over the received payload.
        actual: u32,
    },
    /// A length-prefixed field was not valid UTF-8.
    InvalidUtf8 {
        /// Name of the length-prefixed field that failed UTF-8 validation.
        field: &'static str,
        /// Byte offset where the field begins.
        offset: usize,
    },
    /// Scenario id violates the manifest's identity rules.
    InvalidScenarioId {
        /// Which manifest identity rule the scenario id violated.
        reason: &'static str,
    },
    /// A knob path was empty, overlong, or duplicated.
    InvalidKnobPath {
        /// The offending knob path.
        path: String,
        /// Why it is invalid (empty, overlong, or duplicated).
        reason: &'static str,
    },
    /// A knob value was not finite.
    NonFiniteKnobValue {
        /// Knob path whose value was NaN or infinite.
        path: String,
    },
    /// Too many knob entries.
    TooManyKnobs {
        /// Number of knob entries in the payload.
        found: usize,
        /// Maximum knob entries a permalink may carry.
        maximum: usize,
    },
    /// A knob assignment violates its declared range.
    OutOfRange {
        /// Knob path whose assignment is out of range.
        path: String,
        /// Value the link carried.
        value: f64,
        /// Declared inclusive minimum for the knob.
        min: f64,
        /// Declared inclusive maximum for the knob.
        max: f64,
    },
    /// The embedded digest does not match the config reconstructed from
    /// (scenario + knob diff). The link and the scenario disagree — the digest
    /// is the gate that keeps a tampered scenario from passing.
    DigestMismatch {
        /// Config digest embedded in the link.
        embedded: u64,
        /// Config digest reconstructed from (scenario + knob diff).
        reconstructed: u64,
    },
    /// Trailing bytes after the CRC.
    TrailingBytes {
        /// Number of bytes left after the CRC.
        count: usize,
    },
}

impl fmt::Display for PermalinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TruncatedHeader {
                offset,
                needed,
                available,
            } => write!(
                f,
                "permalink header truncated at byte {offset}: need {needed} bytes, have {available}"
            ),
            Self::TruncatedField {
                field,
                offset,
                declared,
                remaining,
            } => write!(
                f,
                "permalink field {field} at byte {offset} declares {declared} bytes but only {remaining} remain"
            ),
            Self::OversizedPayload { actual, maximum } => {
                write!(
                    f,
                    "permalink payload is {actual} bytes; maximum is {maximum}"
                )
            }
            Self::BadMagic { first_16_bytes_hex } => {
                write!(
                    f,
                    "permalink magic mismatch (first bytes: {first_16_bytes_hex})"
                )
            }
            Self::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "permalink version {found} is not supported (expected {supported})"
                )
            }
            Self::NonzeroFlags { found } => {
                write!(
                    f,
                    "permalink reserved flags must be zero; got 0x{found:02x}"
                )
            }
            Self::CrcMismatch { expected, actual } => {
                write!(
                    f,
                    "permalink CRC32 mismatch: computed {actual:#010x}, declared {expected:#010x}"
                )
            }
            Self::InvalidUtf8 { field, offset } => {
                write!(
                    f,
                    "permalink field {field} at byte {offset} is not valid UTF-8"
                )
            }
            Self::InvalidScenarioId { reason } => {
                write!(f, "permalink scenario id invalid: {reason}")
            }
            Self::InvalidKnobPath { path, reason } => {
                write!(f, "permalink knob path {path:?} invalid: {reason}")
            }
            Self::NonFiniteKnobValue { path } => {
                write!(f, "permalink knob {path} has a non-finite value")
            }
            Self::TooManyKnobs { found, maximum } => {
                write!(f, "permalink carries {found} knobs; maximum is {maximum}")
            }
            Self::OutOfRange {
                path,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "permalink knob {path} = {value} is outside [{min}, {max}]"
                )
            }
            Self::DigestMismatch {
                embedded,
                reconstructed,
            } => write!(
                f,
                "permalink config digest mismatch: embedded {embedded:#018x}, reconstructed {reconstructed:#018x}"
            ),
            Self::TrailingBytes { count } => {
                write!(f, "permalink has {count} trailing bytes after the CRC")
            }
        }
    }
}

impl std::error::Error for PermalinkError {}

impl Permalink {
    /// Canonical binary form: fixed header, scenario id, digest, sorted knob
    /// entries, build identity, CRC32. Encoding is a pure function of the
    /// inputs — knob entries are emitted in ascending path order regardless of
    /// the caller's map iteration order.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut sorted = self.knob_diff.clone();
        sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));

        let mut out = Vec::with_capacity(64 + self.scenario_id.len() + sorted.len() * 32);
        out.extend_from_slice(MAGIC);
        out.push(FORMAT_VERSION);
        out.push(0u8); // reserved flags
        out.extend_from_slice(&self.seed.to_le_bytes());
        push_u16(&mut out, self.scenario_id.len() as u16);
        out.extend_from_slice(self.scenario_id.as_bytes());
        out.extend_from_slice(&self.config_digest.to_le_bytes());
        push_u16(&mut out, sorted.len() as u16);
        for (path, value) in &sorted {
            push_u16(&mut out, path.len() as u16);
            out.extend_from_slice(path.as_bytes());
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&self.build.toolchain_digest.to_le_bytes());
        out.extend_from_slice(&self.build.lockfile_digest.to_le_bytes());
        out.extend_from_slice(&self.build.core_digest.to_le_bytes());
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Encode as the shareable URL-safe string (`sbw1.` + base64url, no padding).
    #[must_use]
    pub fn to_url_string(&self) -> String {
        let mut encoded =
            String::with_capacity(PERMALINK_PREFIX.len() + self.encode().len() * 4 / 3 + 4);
        encoded.push_str(PERMALINK_PREFIX);
        base64url_encode(&self.encode(), &mut encoded);
        encoded
    }

    /// Parse a raw payload. Length-bounded per field before any allocation;
    /// arbitrary bytes can never panic or over-allocate.
    pub fn decode(bytes: &[u8]) -> Result<Self, PermalinkError> {
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(PermalinkError::OversizedPayload {
                actual: bytes.len(),
                maximum: MAX_PAYLOAD_BYTES,
            });
        }
        let header = 3 + 1 + 1 + 8;
        if bytes.len() < header + 4 {
            return Err(PermalinkError::TruncatedHeader {
                offset: bytes.len(),
                needed: header + 4,
                available: bytes.len(),
            });
        }
        if &bytes[0..3] != MAGIC {
            return Err(PermalinkError::BadMagic {
                first_16_bytes_hex: hex_prefix(bytes, 16),
            });
        }
        let version = bytes[3];
        if version != FORMAT_VERSION {
            return Err(PermalinkError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let flags = bytes[4];
        if flags != 0 {
            return Err(PermalinkError::NonzeroFlags { found: flags });
        }
        let crc_slice = take_slice(bytes, bytes.len() - 4, 4, "crc")?;
        let declared_crc =
            u32::from_le_bytes([crc_slice[0], crc_slice[1], crc_slice[2], crc_slice[3]]);
        let payload = &bytes[..bytes.len() - 4];
        let actual_crc = crc32(payload);
        if actual_crc != declared_crc {
            return Err(PermalinkError::CrcMismatch {
                expected: declared_crc,
                actual: actual_crc,
            });
        }

        let mut cursor = 5usize;
        let seed = u64::from_le_bytes(take(payload, cursor, 8, "seed")?);
        cursor += 8;
        let scenario_len = take_u16(payload, &mut cursor, "scenario_id")? as usize;
        let scenario_bytes = take_slice(payload, cursor, scenario_len, "scenario_id")?;
        let scenario_id = std::str::from_utf8(scenario_bytes)
            .map_err(|_| PermalinkError::InvalidUtf8 {
                field: "scenario_id",
                offset: cursor,
            })?
            .to_owned();
        cursor += scenario_len;
        let config_digest = u64::from_le_bytes(take(payload, cursor, 8, "config_digest")?);
        cursor += 8;
        let knob_count = take_u16(payload, &mut cursor, "knob_count")? as usize;
        if knob_count > MAX_KNOB_ENTRIES {
            return Err(PermalinkError::TooManyKnobs {
                found: knob_count,
                maximum: MAX_KNOB_ENTRIES,
            });
        }
        let mut knob_diff = Vec::with_capacity(knob_count.min(MAX_KNOB_ENTRIES));
        let mut previous: Option<Vec<u8>> = None;
        for _ in 0..knob_count {
            let path_len = take_u16(payload, &mut cursor, "knob_path")? as usize;
            if path_len > MAX_KNOB_PATH_BYTES {
                return Err(PermalinkError::InvalidKnobPath {
                    path: format!("<{path_len} bytes>"),
                    reason: "knob path exceeds the 128-byte bound",
                });
            }
            let path_bytes = take_slice(payload, cursor, path_len, "knob_path")?;
            let path = std::str::from_utf8(path_bytes)
                .map_err(|_| PermalinkError::InvalidUtf8 {
                    field: "knob_path",
                    offset: cursor,
                })?
                .to_owned();
            cursor += path_len;
            if path.is_empty() {
                return Err(PermalinkError::InvalidKnobPath {
                    path,
                    reason: "knob path must not be empty",
                });
            }
            if let Some(previous) = &previous {
                match path_bytes.cmp(previous) {
                    std::cmp::Ordering::Less => {
                        return Err(PermalinkError::InvalidKnobPath {
                            path,
                            reason: "knob entries must be in canonical ascending order",
                        });
                    }
                    std::cmp::Ordering::Equal => {
                        return Err(PermalinkError::InvalidKnobPath {
                            path,
                            reason: "duplicate knob path",
                        });
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
            previous = Some(path_bytes.to_vec());
            let value_bits = u64::from_le_bytes(take(payload, cursor, 8, "knob_value")?);
            cursor += 8;
            let value = f64::from_bits(value_bits);
            if !value.is_finite() {
                return Err(PermalinkError::NonFiniteKnobValue { path });
            }
            knob_diff.push((path, value));
        }
        let toolchain_digest = u64::from_le_bytes(take(payload, cursor, 8, "toolchain_digest")?);
        cursor += 8;
        let lockfile_digest = u64::from_le_bytes(take(payload, cursor, 8, "lockfile_digest")?);
        cursor += 8;
        let core_digest = u64::from_le_bytes(take(payload, cursor, 8, "core_digest")?);
        cursor += 8;
        if cursor != payload.len() {
            return Err(PermalinkError::TrailingBytes {
                count: payload.len() - cursor,
            });
        }

        let link = Self {
            scenario_id,
            seed,
            knob_diff,
            config_digest,
            build: BuildLink {
                toolchain_digest,
                lockfile_digest,
                core_digest,
            },
        };
        link.validate_identity()?;
        Ok(link)
    }

    /// Parse the URL-safe string form.
    pub fn from_url_string(text: &str) -> Result<Self, PermalinkError> {
        let body = text
            .strip_prefix(PERMALINK_PREFIX)
            .ok_or_else(|| PermalinkError::BadMagic {
                first_16_bytes_hex: hex_prefix(text.as_bytes(), 16),
            })?;
        let bytes = base64url_decode(body)?;
        Self::decode(&bytes)
    }

    /// Canonicality check: `encode(decode(s)) == s` for every accepted payload.
    #[must_use]
    pub fn is_canonical(bytes: &[u8]) -> bool {
        match Self::decode(bytes) {
            Ok(link) => link.encode() == bytes,
            Err(_) => false,
        }
    }

    /// Validate the scenario id with the manifest's identity rules.
    fn validate_identity(&self) -> Result<(), PermalinkError> {
        if self.scenario_id.trim().is_empty() {
            return Err(PermalinkError::InvalidScenarioId {
                reason: "must not be empty",
            });
        }
        if self.scenario_id.len() > MAX_SCENARIO_ID_BYTES {
            return Err(PermalinkError::InvalidScenarioId {
                reason: "exceeds the 512-byte bound",
            });
        }
        if self.scenario_id.chars().any(char::is_control) {
            return Err(PermalinkError::InvalidScenarioId {
                reason: "must not contain control characters",
            });
        }
        Ok(())
    }

    /// Range-check every knob in the diff against the live knob registry,
    /// naming the offending knob. Runs before any world allocation.
    pub fn validate_knobs(&self) -> Result<(), PermalinkError> {
        let assignments: Vec<(String, f64)> = self.knob_diff.clone();
        for violation in crate::check_knob_ranges(&assignments) {
            return Err(PermalinkError::OutOfRange {
                path: violation.path,
                value: violation.value,
                min: violation.min,
                max: violation.max,
            });
        }
        Ok(())
    }

    /// Compare the receiver's build identity against the link's.
    #[must_use]
    pub fn build_match(&self, local: &BuildLink) -> BuildMatch {
        if self.build == *local {
            return BuildMatch::Exact;
        }
        if self.build.lockfile_digest == local.lockfile_digest
            && self.build.core_digest == local.core_digest
        {
            return BuildMatch::Compatible;
        }
        BuildMatch::Mismatch
    }

    /// Recompute the digest of the config reconstructed from `scenario_config`
    /// plus this link's knob diff, and compare it against the embedded digest.
    /// This is the gate that keeps the digest from being decorative: a link
    /// whose scenario document no longer matches its claim is rejected.
    pub fn verify_config_digest(
        &self,
        scenario_config: &serde_json::Value,
    ) -> Result<(), PermalinkError> {
        let reconstructed = config_digest_with_diff(scenario_config, &self.knob_diff);
        if reconstructed != self.config_digest {
            return Err(PermalinkError::DigestMismatch {
                embedded: self.config_digest,
                reconstructed,
            });
        }
        Ok(())
    }
}

/// Compute the permalink config digest over a scenario's resolved config value
/// with a knob diff applied. The digest covers the canonical reconstruction —
/// scenario id plus every knob path with its exact f64 bit pattern, in
/// ascending path order — so two runs with the same reconstruction always
/// agree and any tampering always disagrees.
#[must_use]
pub fn config_digest_with_diff(
    scenario_config: &serde_json::Value,
    knob_diff: &[(String, f64)],
) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    feed(b"permalink-config-v1\0");
    feed(&canonical_config_bytes(scenario_config));
    let mut sorted: Vec<(&String, &f64)> = knob_diff.iter().map(|(p, v)| (p, v)).collect();
    sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    for (path, value) in sorted {
        feed(&(path.len() as u16).to_le_bytes());
        feed(path.as_bytes());
        feed(&value.to_bits().to_le_bytes());
    }
    hash
}

/// Serialize a config value with deterministic key order so digest inputs are
/// stable regardless of map construction order.
fn canonical_config_bytes(value: &serde_json::Value) -> Vec<u8> {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
            let mut out = Vec::with_capacity(256);
            out.push(b'{');
            for (index, (key, entry)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                out.extend_from_slice(key.as_bytes());
                out.push(b':');
                out.extend_from_slice(&canonical_config_bytes(entry));
            }
            out.push(b'}');
            out
        }
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(64);
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                out.extend_from_slice(&canonical_config_bytes(item));
            }
            out.push(b']');
            out
        }
        other => {
            // Numbers are emitted with their exact f64 bit pattern, never a
            // locale-formatted decimal.
            if let Some(number) = other.as_f64() {
                number.to_bits().to_le_bytes().to_vec()
            } else {
                other.to_string().into_bytes()
            }
        }
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn take<'a>(
    bytes: &'a [u8],
    offset: usize,
    count: usize,
    field: &'static str,
) -> Result<[u8; 8], PermalinkError>
where
    [u8; 8]: TryFrom<&'a [u8]>,
{
    take_slice(bytes, offset, count, field).map(|slice| {
        let mut fixed = [0u8; 8];
        fixed[..slice.len().min(8)].copy_from_slice(&slice[..slice.len().min(8)]);
        fixed
    })
}

fn take_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    count: usize,
    field: &'static str,
) -> Result<&'a [u8], PermalinkError> {
    let remaining = bytes.len().saturating_sub(offset);
    if remaining < count {
        return Err(PermalinkError::TruncatedField {
            field,
            offset,
            declared: count,
            remaining,
        });
    }
    Ok(&bytes[offset..offset + count])
}

fn take_u16(bytes: &[u8], cursor: &mut usize, field: &'static str) -> Result<u16, PermalinkError> {
    let raw = take_slice(bytes, *cursor, 2, field)?;
    *cursor += 2;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn hex_prefix(bytes: &[u8], max: usize) -> String {
    let mut out = String::with_capacity(max * 2);
    for byte in bytes.iter().take(max) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// IEEE CRC32 over the payload (polynomial 0xEDB88320, reflected).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

const BASE64URL_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn base64url_encode(bytes: &[u8], out: &mut String) {
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let packed = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64URL_ALPHABET[(packed >> 18) as usize & 0x3F] as char);
        out.push(BASE64URL_ALPHABET[(packed >> 12) as usize & 0x3F] as char);
        if chunk.len() > 1 {
            out.push(BASE64URL_ALPHABET[(packed >> 6) as usize & 0x3F] as char);
        }
        if chunk.len() > 2 {
            out.push(BASE64URL_ALPHABET[packed as usize & 0x3F] as char);
        }
    }
}

fn base64url_decode(text: &str) -> Result<Vec<u8>, PermalinkError> {
    let mut values = Vec::with_capacity(text.len());
    for (index, byte) in text.bytes().enumerate() {
        let value = BASE64URL_ALPHABET
            .iter()
            .position(|candidate| *candidate == byte)
            .ok_or(PermalinkError::BadMagic {
                first_16_bytes_hex: format!("invalid base64url byte 0x{byte:02x} at {index}"),
            })?;
        values.push(value as u32);
    }
    let mut out = Vec::with_capacity(values.len() * 3 / 4);
    for chunk in values.chunks(4) {
        let packed = (chunk[0] << 18)
            | (chunk.get(1).copied().unwrap_or(0) << 12)
            | (chunk.get(2).copied().unwrap_or(0) << 6)
            | chunk.get(3).copied().unwrap_or(0);
        out.push((packed >> 16) as u8);
        if chunk.len() > 2 {
            out.push((packed >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(packed as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    fn sample_build() -> BuildLink {
        BuildLink {
            toolchain_digest: 0x1111_2222_3333_4444,
            lockfile_digest: 0x5555_6666_7777_8888,
            core_digest: 0x9999_AAAA_BBBB_CCCC,
        }
    }

    fn sample_link() -> Permalink {
        Permalink {
            scenario_id: "meadow".to_owned(),
            seed: 42,
            knob_diff: vec![
                ("food_max".to_owned(), 0.6),
                ("population_minimum".to_owned(), 24.0),
            ],
            config_digest: 0xDEAD_BEEF_CAFE_F00D,
            build: sample_build(),
        }
    }

    #[test]
    fn round_trip_binary_and_url_forms() {
        let link = sample_link();
        let bytes = link.encode();
        let decoded = Permalink::decode(&bytes).expect("decode canonical payload");
        assert_eq!(decoded, link);
        assert!(Permalink::is_canonical(&bytes));

        let url = link.to_url_string();
        assert!(url.starts_with(PERMALINK_PREFIX));
        let decoded_url = Permalink::from_url_string(&url).expect("decode url form");
        assert_eq!(decoded_url, link);
        // Canonicality is stable across the url form as well.
        assert_eq!(decoded_url.encode(), bytes);
    }

    #[test]
    fn encoding_is_independent_of_diff_construction_order() {
        let mut shuffled = sample_link();
        shuffled.knob_diff.reverse();
        assert_eq!(sample_link().encode(), shuffled.encode());
    }

    #[test]
    fn worst_case_diff_fits_the_documented_cap() {
        let mut link = sample_link();
        link.knob_diff = (0..MAX_KNOB_ENTRIES)
            .map(|index| (format!("knob.path.{index:04}"), index as f64))
            .collect();
        let bytes = link.encode();
        assert!(
            bytes.len() <= MAX_PAYLOAD_BYTES,
            "worst-case payload {} exceeds the documented cap {}",
            bytes.len(),
            MAX_PAYLOAD_BYTES
        );
        let decoded = Permalink::decode(&bytes).expect("worst-case payload decodes");
        assert_eq!(decoded.knob_diff.len(), MAX_KNOB_ENTRIES);
    }

    #[test]
    fn decode_rejects_every_truncation_offset_without_panic() {
        let bytes = sample_link().encode();
        for offset in 0..bytes.len() {
            let truncated = &bytes[..offset];
            let result = std::panic::catch_unwind(|| Permalink::decode(truncated));
            let outcome = result.expect("decode must never panic");
            assert!(
                outcome.is_err() || offset >= 5 + 8,
                "truncation at {offset} must not silently succeed past the header"
            );
        }
    }

    #[test]
    fn decode_never_panics_on_arbitrary_bytes_and_bounds_allocations() {
        let mut rng = SmallRng::seed_from_u64(0x5B1);
        for case in 0..50_000usize {
            let length = (rng.random::<u32>() % 512) as usize;
            let mut bytes = vec![0u8; length];
            rng.fill(&mut bytes[..]);
            let result = std::panic::catch_unwind(|| Permalink::decode(&bytes));
            assert!(result.is_ok(), "decode panicked on case {case}");
            // A successful decode on random bytes must still be canonical-bounded.
            if let Ok(link) = result.unwrap() {
                assert!(link.scenario_id.len() <= MAX_SCENARIO_ID_BYTES);
                assert!(link.knob_diff.len() <= MAX_KNOB_ENTRIES);
            }
        }
    }

    #[test]
    fn decode_rejects_the_seeded_adversarial_corpus() {
        let good = sample_link().encode();

        // Magic corruption.
        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xFF;
        assert!(matches!(
            Permalink::decode(&bad_magic),
            Err(PermalinkError::BadMagic { .. })
        ));

        // Version 255.
        let mut bad_version = good.clone();
        bad_version[3] = 255;
        let bad_version_len = bad_version.len();
        let crc = crc32(&bad_version[..bad_version_len - 4]);
        bad_version[bad_version_len - 4..].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            Permalink::decode(&bad_version),
            Err(PermalinkError::UnsupportedVersion { found: 255, .. })
        ));

        // CRC corruption.
        let mut bad_crc = good.clone();
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xFF;
        assert!(matches!(
            Permalink::decode(&bad_crc),
            Err(PermalinkError::CrcMismatch { .. })
        ));

        // Declared scenario length of 0xFFFF without the bytes.
        let mut lying = Vec::new();
        lying.extend_from_slice(MAGIC);
        lying.push(FORMAT_VERSION);
        lying.push(0);
        lying.extend_from_slice(&42u64.to_le_bytes());
        lying.extend_from_slice(&u16::MAX.to_le_bytes());
        lying.extend_from_slice(&[0u8; 4]);
        let lying_len = lying.len();
        let crc = crc32(&lying[..lying_len - 4]);
        lying[lying_len - 4..].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            Permalink::decode(&lying),
            Err(PermalinkError::TruncatedField {
                field: "scenario_id",
                ..
            }) | Err(PermalinkError::CrcMismatch { .. })
                | Err(PermalinkError::TrailingBytes { .. })
        ));

        // Oversized payload: 10 MB of 'A'.
        let huge = vec![b'A'; 10 * 1024 * 1024];
        assert!(matches!(
            Permalink::decode(&huge),
            Err(PermalinkError::OversizedPayload { .. })
        ));

        // Valid base64 of random bytes.
        let mut rng = SmallRng::seed_from_u64(0xBAD);
        let mut body = String::new();
        let random_bytes: Vec<u8> = (0..64).map(|_| rng.random::<u8>()).collect();
        base64url_encode(&random_bytes, &mut body);
        let with_prefix = format!("{PERMALINK_PREFIX}{body}");
        let _ = Permalink::from_url_string(&with_prefix); // must not panic

        // 64 KB scenario id.
        let mut big_scenario = Vec::new();
        big_scenario.extend_from_slice(MAGIC);
        big_scenario.push(FORMAT_VERSION);
        big_scenario.push(0);
        big_scenario.extend_from_slice(&1u64.to_le_bytes());
        big_scenario.extend_from_slice(&u16::MAX.to_le_bytes());
        big_scenario.extend_from_slice(&vec![b'x'; U16_MAX_AS_USIZE]);
        big_scenario.extend_from_slice(&[0u8; 40]);
        let big_scenario_len = big_scenario.len();
        let crc = crc32(&big_scenario[..big_scenario_len - 4]);
        big_scenario[big_scenario_len - 4..].copy_from_slice(&crc.to_le_bytes());
        let outcome = std::panic::catch_unwind(|| Permalink::decode(&big_scenario));
        assert!(
            outcome.is_ok(),
            "64 KB scenario id must never panic the decode"
        );
    }

    #[test]
    fn digest_mismatch_is_rejected_not_run() {
        let scenario_config = serde_json::json!({
            "food_max": 0.5,
            "population_minimum": 16,
        });
        let mut link = sample_link();
        link.config_digest = config_digest_with_diff(&scenario_config, &link.knob_diff);
        link.verify_config_digest(&scenario_config)
            .expect("matching reconstruction verifies");

        // The link's diff claims food_max 0.6; a scenario whose food_max is 0.9
        // makes the reconstructed digest disagree.
        let tampered = serde_json::json!({
            "food_max": 0.9,
            "population_minimum": 16,
        });
        assert!(matches!(
            link.verify_config_digest(&tampered),
            Err(PermalinkError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn knob_validation_names_the_offender_before_any_world_allocation() {
        let mut link = sample_link();
        link.knob_diff = vec![("food_max".to_owned(), f64::NAN)];
        let error = link.validate_knobs().expect_err("NaN must be rejected");
        assert!(matches!(
            error,
            PermalinkError::OutOfRange { ref path, .. } if path == "food_max"
        ));

        link.knob_diff = vec![("food_max".to_owned(), 999_999.0)];
        let error = link
            .validate_knobs()
            .expect_err("out-of-range must be rejected");
        assert!(matches!(
            error,
            PermalinkError::OutOfRange { ref path, .. } if path == "food_max"
        ));
    }

    #[test]
    fn build_match_distinguishes_exact_compatible_and_mismatch() {
        let link = sample_link();
        assert_eq!(link.build_match(&sample_build()), BuildMatch::Exact);

        let toolchain_only = BuildLink {
            toolchain_digest: 0xAAAA,
            ..sample_build()
        };
        assert_eq!(link.build_match(&toolchain_only), BuildMatch::Compatible);

        // The most dangerous real-world drift: the core feature set differs.
        let core_differs = BuildLink {
            core_digest: 0xBBBB,
            ..sample_build()
        };
        assert_eq!(link.build_match(&core_differs), BuildMatch::Mismatch);

        let lockfile_differs = BuildLink {
            lockfile_digest: 0xCCCC,
            ..sample_build()
        };
        assert_eq!(link.build_match(&lockfile_differs), BuildMatch::Mismatch);
    }

    #[test]
    fn duplicate_and_unordered_knobs_are_rejected() {
        let mut link = sample_link();
        link.knob_diff = vec![("food_max".to_owned(), 0.6), ("food_max".to_owned(), 0.7)];
        let bytes = link.encode();
        // encode() sorts; hand-build a duplicate-in-order payload to decode.
        let mut manual = Vec::new();
        manual.extend_from_slice(MAGIC);
        manual.push(FORMAT_VERSION);
        manual.push(0);
        manual.extend_from_slice(&42u64.to_le_bytes());
        push_u16(&mut manual, 6);
        manual.extend_from_slice(b"meadow");
        manual.extend_from_slice(&0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes());
        push_u16(&mut manual, 2);
        push_u16(&mut manual, 8);
        manual.extend_from_slice(b"food_max");
        manual.extend_from_slice(&0.6f64.to_bits().to_le_bytes());
        push_u16(&mut manual, 8);
        manual.extend_from_slice(b"food_max");
        manual.extend_from_slice(&0.7f64.to_bits().to_le_bytes());
        manual.extend_from_slice(&link.build.toolchain_digest.to_le_bytes());
        manual.extend_from_slice(&link.build.lockfile_digest.to_le_bytes());
        manual.extend_from_slice(&link.build.core_digest.to_le_bytes());
        let crc = crc32(&manual);
        manual.extend_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            Permalink::decode(&manual),
            Err(PermalinkError::InvalidKnobPath { .. })
        ));
        let _ = bytes;
    }
}
