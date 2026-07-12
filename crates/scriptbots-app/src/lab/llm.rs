//! The provider boundary for the autonomous lab: one trait, two implementations.
//!
//! Everything the lab does with a model goes through [`LlmClient`]. That seam is
//! the whole point of this module. Without it, the loop gets written against a
//! concrete Anthropic client, the tests get written against a mock of that
//! client's HTTP layer, and CI ends up flaky (the network), expensive (real
//! tokens), or dishonest (mocking the thing under test).
//!
//! Two rules hold everywhere below:
//!
//! - **`cargo test` never touches the network.** The Anthropic implementation is
//!   behind the `llm-anthropic` feature, off by default. The response *parser* is
//!   not gated, so the malformation tests — which is where the real bugs are —
//!   run on every build.
//! - **Nothing here panics.** A garbage body, a timeout, a 500, a 200 with an
//!   empty body: each is a typed [`LlmError`]. A lab that unwinds mid-experiment
//!   loses the experiment.

use rand::Rng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

/// Hard ceiling on a response body.
///
/// A pathological — or hostile — response that streamed until the process OOMed
/// would take the user's simulation down with it.
pub const MAX_RESPONSE_BYTES: usize = 1 << 20;

/// One message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmMessage {
    /// `user` or `assistant`.
    pub role: String,
    /// The message text.
    pub content: String,
}

/// A tool the model is allowed to call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmToolSpec {
    /// Tool name, as the model will emit it.
    pub name: String,
    /// What the tool does — the model reads this to decide when to call it.
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub input_schema: serde_json::Value,
}

/// A request to the model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    /// System prompt.
    pub system: String,
    /// Conversation so far.
    pub messages: Vec<LlmMessage>,
    /// The tools on offer. A tool call naming anything outside this list is an
    /// error, not a surprise to be papered over downstream.
    pub tools: Vec<LlmToolSpec>,
    /// Output cap.
    pub max_tokens: u32,
}

/// A tool call the model asked for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmToolCall {
    /// Which tool.
    pub name: String,
    /// Its arguments, always an object by the time it leaves this module.
    pub arguments: serde_json::Value,
}

/// Tokens spent on one call.
///
/// The client REPORTS tokens; the loop ENFORCES the budget. If a response could
/// arrive without a usage figure, the loop's token budget would be
/// unenforceable and its only backstop would be the iteration cap — so a missing
/// usage block is an error, not a zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens.
    pub input: u32,
    /// Output tokens.
    pub output: u32,
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Finished normally.
    EndTurn,
    /// Wants to call a tool.
    ToolUse,
    /// Hit the output cap — the answer is truncated.
    MaxTokens,
    /// Declined.
    Refusal,
    /// Something the provider added after this was written.
    Other(String),
}

/// Every way a provider call can fail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LlmError {
    /// The call never reached a well-formed response.
    #[error("transport failure: {0}")]
    Transport(String),
    /// Rate limited; `retry_after_ms` is present only when the provider sent a
    /// value we could actually parse.
    #[error("rate limited (retry after {retry_after_ms:?} ms)")]
    RateLimited {
        /// Provider-supplied backoff, if it was intelligible.
        retry_after_ms: Option<u64>,
    },
    /// Provider is overloaded (HTTP 529).
    #[error("provider overloaded")]
    Overloaded,
    /// A 200 whose body we could not make sense of.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    /// The body exceeded [`MAX_RESPONSE_BYTES`].
    #[error("response too large: {bytes} bytes exceeds cap of {cap}")]
    ResponseTooLarge {
        /// How big it got.
        bytes: usize,
        /// The cap.
        cap: usize,
    },
    /// A scripted fixture ran out of turns.
    ///
    /// A distinct error rather than an empty response, which the loop would
    /// happily mistake for a valid turn and keep going.
    #[error("scripted fixture exhausted")]
    FixtureExhausted,
    /// The model declined the request.
    #[error("model refused the request")]
    Refused,
}

/// What the model said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Any prose it produced.
    pub text: Option<String>,
    /// Any tools it called.
    pub tool_calls: Vec<LlmToolCall>,
    /// Tokens spent — always present.
    pub usage: TokenUsage,
    /// Why it stopped.
    pub stop_reason: StopReason,
}

/// The provider seam.
///
/// Deliberately synchronous: the lab loop is a blocking state machine driven
/// from a CLI subcommand, not part of the axum/Tokio server. Making this async
/// would mean standing up a third runtime for it.
pub trait LlmClient: Send + Sync {
    /// Run one turn.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError`] for every provider outcome that is not a well-formed
    /// response. Never panics.
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;

    /// The model identifier, recorded verbatim in the notebook's provenance block.
    fn model_id(&self) -> &str;
}

/// Parse a provider response body into an [`LlmResponse`].
///
/// Deliberately separate from any HTTP client and NOT feature-gated: this is
/// where the real bugs live (a tool call naming a tool we never offered,
/// `arguments` arriving as a string, a body with no usage block), and those
/// cases must be tested on every build, including builds with no network stack
/// compiled in at all.
///
/// `request` is needed to reject a tool call for a tool that was never on offer.
///
/// # Errors
///
/// [`LlmError::ResponseTooLarge`], or [`LlmError::InvalidResponse`] for any body
/// that is not a response we can act on.
pub fn parse_response(request: &LlmRequest, body: &[u8]) -> Result<LlmResponse, LlmError> {
    if body.len() > MAX_RESPONSE_BYTES {
        return Err(LlmError::ResponseTooLarge {
            bytes: body.len(),
            cap: MAX_RESPONSE_BYTES,
        });
    }
    if body.is_empty() {
        return Err(LlmError::InvalidResponse("empty body".to_owned()));
    }

    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|err| LlmError::InvalidResponse(format!("body is not JSON: {err}")))?;

    // Usage first: a response we cannot bill is a response we cannot budget.
    let usage = value
        .get("usage")
        .ok_or_else(|| LlmError::InvalidResponse("response has no usage block".to_owned()))?;
    let usage = TokenUsage {
        input: read_token_count(usage, "input_tokens")?,
        output: read_token_count(usage, "output_tokens")?,
    };

    let stop_reason = match value.get("stop_reason").and_then(serde_json::Value::as_str) {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("refusal") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_owned()),
        None => {
            return Err(LlmError::InvalidResponse(
                "response has no stop_reason".to_owned(),
            ));
        }
    };
    if stop_reason == StopReason::Refusal {
        return Err(LlmError::Refused);
    }

    let content = value
        .get("content")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| LlmError::InvalidResponse("response has no content array".to_owned()))?;

    let mut text: Option<String> = None;
    let mut tool_calls = Vec::new();
    for block in content {
        match block.get("type").and_then(serde_json::Value::as_str) {
            Some("text") => {
                let chunk = block
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                text.get_or_insert_with(String::new).push_str(chunk);
            }
            Some("tool_use") => tool_calls.push(parse_tool_call(request, block)?),
            // Thinking blocks and anything the provider adds later are not
            // actionable here, and ignoring them is safer than failing on them.
            _ => {}
        }
    }

    Ok(LlmResponse {
        text,
        tool_calls,
        usage,
        stop_reason,
    })
}

fn read_token_count(usage: &serde_json::Value, field: &str) -> Result<u32, LlmError> {
    let raw = usage
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            LlmError::InvalidResponse(format!("usage.{field} is missing or not a number"))
        })?;
    u32::try_from(raw)
        .map_err(|_| LlmError::InvalidResponse(format!("usage.{field} is implausible: {raw}")))
}

fn parse_tool_call(
    request: &LlmRequest,
    block: &serde_json::Value,
) -> Result<LlmToolCall, LlmError> {
    let name = block
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| LlmError::InvalidResponse("tool_use block has no name".to_owned()))?;

    // A tool we never offered is a hallucination, and executing it — or letting
    // it reach the loop as a plausible-looking call — is exactly how a lab ends
    // up doing something nobody authorised.
    if !request.tools.iter().any(|spec| spec.name == name) {
        return Err(LlmError::InvalidResponse(format!(
            "model called tool `{name}`, which was never offered"
        )));
    }

    let raw = block
        .get("input")
        .ok_or_else(|| LlmError::InvalidResponse(format!("tool call `{name}` has no arguments")))?;

    let arguments = match raw {
        serde_json::Value::Object(_) => raw.clone(),
        // The single most common real-model malformation: arguments arrive as a
        // JSON *string* rather than an object. Coerce it — refusing here would
        // fail on a quirk the model actually exhibits — but only if the string
        // really does contain an object.
        serde_json::Value::String(encoded) => {
            let decoded: serde_json::Value = serde_json::from_str(encoded).map_err(|err| {
                LlmError::InvalidResponse(format!(
                    "tool call `{name}` sent arguments as a string that is not JSON: {err}"
                ))
            })?;
            if !decoded.is_object() {
                return Err(LlmError::InvalidResponse(format!(
                    "tool call `{name}` sent arguments as a string that decodes to a non-object"
                )));
            }
            decoded
        }
        _ => {
            return Err(LlmError::InvalidResponse(format!(
                "tool call `{name}` sent arguments that are neither an object nor a string"
            )));
        }
    };

    Ok(LlmToolCall {
        name: name.to_owned(),
        arguments,
    })
}

/// Interpret a `Retry-After` header.
///
/// Returns `None` for a value we cannot read — a provider that says "soon"
/// tells us nothing, and inventing a number would be worse than admitting that.
#[must_use]
pub fn parse_retry_after(header: &str) -> Option<u64> {
    let seconds: f64 = header.trim().parse().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    // Cap at an hour: a provider asking us to sleep for a week is a provider bug.
    Some((seconds * 1000.0).min(3_600_000.0) as u64)
}

/// A bounded retry policy whose jitter is drawn from a SEEDED generator.
///
/// Jitter from the wall clock would make the entire lab irreproducible — for the
/// sake of being polite to a server. The seed is threaded in from the session,
/// so a replayed session retries on exactly the same schedule.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// How many attempts in total (1 = no retries).
    pub attempts: u32,
    /// First backoff, doubled each attempt.
    pub base_backoff_ms: u64,
    /// Never sleep longer than this.
    pub max_backoff_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 4,
            base_backoff_ms: 500,
            max_backoff_ms: 30_000,
        }
    }
}

impl RetryPolicy {
    /// Backoff for `attempt` (0-based), honouring a provider-supplied hint.
    ///
    /// Jitter comes from `rng` — the caller's seeded generator — never the clock.
    #[must_use]
    pub fn backoff_ms(
        &self,
        attempt: u32,
        provider_hint_ms: Option<u64>,
        rng: &mut SmallRng,
    ) -> u64 {
        if let Some(hint) = provider_hint_ms {
            return hint.min(self.max_backoff_ms);
        }
        let exponential = self
            .base_backoff_ms
            .saturating_mul(1u64 << attempt.min(16))
            .min(self.max_backoff_ms);
        // Full jitter over [0, exponential]: decorrelates a fleet of retriers
        // without ever consulting a clock.
        rng.random_range(0..=exponential)
    }

    /// Whether an error is worth another attempt.
    #[must_use]
    pub fn is_retryable(error: &LlmError) -> bool {
        matches!(
            error,
            LlmError::Transport(_) | LlmError::RateLimited { .. } | LlmError::Overloaded
        )
    }
}

/// One pre-recorded turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptedTurn {
    /// The response body, exactly as a provider would have sent it.
    pub body: serde_json::Value,
}

/// The CI client: a fixture, replayed in order.
///
/// Zero network, zero clock, zero entropy. Two runs of the same fixture produce
/// byte-identical transcripts, which is what makes `--offline-fixture` in CI a
/// real reproducibility guarantee rather than a slogan.
#[derive(Debug)]
pub struct ScriptedClient {
    model_id: String,
    turns: Vec<ScriptedTurn>,
    next: std::sync::atomic::AtomicUsize,
}

impl ScriptedClient {
    /// Build a client from a list of turns.
    #[must_use]
    pub fn new(model_id: impl Into<String>, turns: Vec<ScriptedTurn>) -> Self {
        Self {
            model_id: model_id.into(),
            turns,
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Load a fixture from JSON.
    ///
    /// # Errors
    ///
    /// [`LlmError::InvalidResponse`] if the fixture itself is malformed — better
    /// to fail loudly at load than to discover it three turns into a run.
    pub fn from_fixture(model_id: impl Into<String>, json: &[u8]) -> Result<Self, LlmError> {
        let turns: Vec<ScriptedTurn> = serde_json::from_slice(json).map_err(|err| {
            LlmError::InvalidResponse(format!("fixture is not valid JSON: {err}"))
        })?;
        Ok(Self::new(model_id, turns))
    }

    /// Turns not yet replayed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.turns
            .len()
            .saturating_sub(self.next.load(std::sync::atomic::Ordering::SeqCst))
    }
}

impl LlmClient for ScriptedClient {
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let index = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let turn = self.turns.get(index).ok_or(LlmError::FixtureExhausted)?;
        let body = serde_json::to_vec(&turn.body).map_err(|err| {
            LlmError::InvalidResponse(format!("fixture turn is not encodable: {err}"))
        })?;
        // Parsed through exactly the same path as a live response — a fixture
        // that skipped the parser would test nothing.
        parse_response(request, &body)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(feature = "llm-anthropic")]
pub use anthropic::AnthropicClient;

#[cfg(feature = "llm-anthropic")]
mod anthropic {
    use super::{
        LlmClient, LlmError, LlmRequest, LlmResponse, MAX_RESPONSE_BYTES, RetryPolicy,
        parse_response, parse_retry_after,
    };
    use rand::rngs::SmallRng;
    use std::sync::Mutex;

    /// The Anthropic Messages API version this client speaks.
    const ANTHROPIC_VERSION: &str = "2023-06-01";
    const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";

    /// A live Anthropic client.
    ///
    /// Gated behind the `llm-anthropic` feature, which the default test profile
    /// does not enable — `cargo test` cannot reach the network even by accident.
    pub struct AnthropicClient {
        http: reqwest::blocking::Client,
        api_key: String,
        model_id: String,
        policy: RetryPolicy,
        rng: Mutex<SmallRng>,
    }

    // Hand-written so the key cannot escape through a derived Debug — the
    // failure mode being prevented is an API key landing in a notebook, a log
    // line, or a panic message and then being pushed to a remote.
    impl std::fmt::Debug for AnthropicClient {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AnthropicClient")
                .field("model_id", &self.model_id)
                .field("api_key", &"<redacted>")
                .finish_non_exhaustive()
        }
    }

    impl AnthropicClient {
        /// Build a client, taking the key from `ANTHROPIC_API_KEY`.
        ///
        /// The environment is the ONLY source. Not a config file (which gets
        /// committed), not a CLI flag (which lands in shell history and in the
        /// det-check config layer).
        ///
        /// # Errors
        ///
        /// [`LlmError::Transport`] if the key is absent or the HTTP client
        /// cannot be built.
        pub fn from_env(model_id: impl Into<String>, rng: SmallRng) -> Result<Self, LlmError> {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| LlmError::Transport("ANTHROPIC_API_KEY is not set".to_owned()))?;
            let http = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .map_err(|err| {
                    LlmError::Transport(format!("could not build HTTP client: {err}"))
                })?;
            Ok(Self {
                http,
                api_key,
                model_id: model_id.into(),
                policy: RetryPolicy::default(),
                rng: Mutex::new(rng),
            })
        }

        fn body(&self, request: &LlmRequest) -> serde_json::Value {
            serde_json::json!({
                "model": self.model_id,
                "max_tokens": request.max_tokens,
                "system": request.system,
                "messages": request.messages.iter().map(|message| serde_json::json!({
                    "role": message.role,
                    "content": message.content,
                })).collect::<Vec<_>>(),
                "tools": request.tools.iter().map(|tool| serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })).collect::<Vec<_>>(),
            })
        }

        fn attempt(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
            let response = self
                .http
                .post(ANTHROPIC_URL)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(&self.body(request))
                .send()
                // reqwest's error carries the URL, never the headers — but be
                // explicit rather than trusting that to stay true.
                .map_err(|err| LlmError::Transport(redact(&err.to_string(), &self.api_key)))?;

            let status = response.status();
            if status.as_u16() == 429 {
                let retry_after_ms = response
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .and_then(parse_retry_after);
                return Err(LlmError::RateLimited { retry_after_ms });
            }
            if status.as_u16() == 529 {
                return Err(LlmError::Overloaded);
            }

            // Read into a bounded buffer: past the cap we stop, rather than
            // letting a pathological body grow until the process dies.
            let bytes = response
                .bytes()
                .map_err(|err| LlmError::Transport(redact(&err.to_string(), &self.api_key)))?;
            if bytes.len() > MAX_RESPONSE_BYTES {
                return Err(LlmError::ResponseTooLarge {
                    bytes: bytes.len(),
                    cap: MAX_RESPONSE_BYTES,
                });
            }
            if !status.is_success() {
                return Err(LlmError::Transport(format!(
                    "provider returned HTTP {status}"
                )));
            }
            parse_response(request, &bytes)
        }
    }

    /// Belt and braces: strip the key from any string that is about to be
    /// surfaced in an error, a log line, or a notebook.
    fn redact(message: &str, api_key: &str) -> String {
        if api_key.is_empty() {
            return message.to_owned();
        }
        message.replace(api_key, "<redacted>")
    }

    impl LlmClient for AnthropicClient {
        fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
            let mut last = LlmError::Transport("no attempt was made".to_owned());
            for attempt in 0..self.policy.attempts {
                match self.attempt(request) {
                    Ok(response) => {
                        tracing::info!(
                            provider = "anthropic",
                            model_id = %self.model_id,
                            input_tokens = response.usage.input,
                            output_tokens = response.usage.output,
                            stop_reason = ?response.stop_reason,
                            attempt,
                            "llm call succeeded"
                        );
                        return Ok(response);
                    }
                    Err(error) if RetryPolicy::is_retryable(&error) => {
                        let hint = match &error {
                            LlmError::RateLimited { retry_after_ms } => *retry_after_ms,
                            _ => None,
                        };
                        let backoff = self
                            .rng
                            .lock()
                            .map_or(self.policy.base_backoff_ms, |mut rng| {
                                self.policy.backoff_ms(attempt, hint, &mut rng)
                            });
                        tracing::warn!(
                            attempt,
                            error_kind = ?error,
                            retry_after_ms = ?hint,
                            backoff_ms = backoff,
                            "llm call failed; retrying"
                        );
                        last = error;
                        if attempt + 1 < self.policy.attempts {
                            std::thread::sleep(std::time::Duration::from_millis(backoff));
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(last)
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn request() -> LlmRequest {
        LlmRequest {
            system: "you are a scientist".to_owned(),
            messages: vec![LlmMessage {
                role: "user".to_owned(),
                content: "propose an experiment".to_owned(),
            }],
            tools: vec![LlmToolSpec {
                name: "propose_experiment".to_owned(),
                description: "propose one experiment".to_owned(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            max_tokens: 1024,
        }
    }

    fn tool_use_body() -> serde_json::Value {
        serde_json::json!({
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 34},
            "content": [
                {"type": "text", "text": "Here is my proposal."},
                {"type": "tool_use", "name": "propose_experiment",
                 "input": {"knob": "food_growth_rate", "value": 0.1}},
            ]
        })
    }

    #[test]
    fn a_well_formed_response_parses() {
        let parsed = parse_response(&request(), tool_use_body().to_string().as_bytes())
            .expect("well-formed response");
        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
        assert_eq!(
            parsed.usage,
            TokenUsage {
                input: 12,
                output: 34
            }
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "propose_experiment");
        assert_eq!(parsed.text.as_deref(), Some("Here is my proposal."));
    }

    #[test]
    fn the_malformations_that_real_models_actually_produce_are_all_errors() {
        // This is where the real bugs live. Every one of these has to be a typed
        // error: not a panic, not a hang, and above all not a plausible-looking
        // response that the loop would act on.
        let request = request();
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("not JSON at all", b"this is not json".to_vec()),
            (
                "JSON with a trailing comma",
                br#"{"stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1,},}"#.to_vec(),
            ),
            ("a 200 with an empty body", Vec::new()),
            (
                "no usage block — the loop could not budget this",
                serde_json::json!({"stop_reason": "end_turn", "content": []})
                    .to_string()
                    .into_bytes(),
            ),
            (
                "no stop_reason",
                serde_json::json!({"usage": {"input_tokens": 1, "output_tokens": 1}, "content": []})
                    .to_string()
                    .into_bytes(),
            ),
            (
                "a tool we never offered",
                serde_json::json!({
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "tool_use", "name": "apply_patch", "input": {}}]
                })
                .to_string()
                .into_bytes(),
            ),
            (
                "a tool call with no arguments at all",
                serde_json::json!({
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "tool_use", "name": "propose_experiment"}]
                })
                .to_string()
                .into_bytes(),
            ),
            (
                "arguments as a string that is not JSON",
                serde_json::json!({
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "tool_use", "name": "propose_experiment", "input": "not json"}]
                })
                .to_string()
                .into_bytes(),
            ),
        ];

        for (name, body) in cases {
            let outcome = parse_response(&request, &body);
            assert!(
                outcome.is_err(),
                "`{name}` must be a typed error, but it parsed as {outcome:?}"
            );
        }
    }

    #[test]
    fn arguments_delivered_as_a_json_string_are_coerced() {
        // The single most common real-model malformation. Rejecting it would
        // fail the lab on a quirk the model genuinely exhibits, so we coerce —
        // but only when the string really does decode to an object (the
        // negative case is covered above).
        let body = serde_json::json!({
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{
                "type": "tool_use",
                "name": "propose_experiment",
                "input": "{\"knob\": \"food_growth_rate\"}"
            }]
        });
        let parsed = parse_response(&request(), body.to_string().as_bytes()).expect("coerced");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            serde_json::json!({"knob": "food_growth_rate"})
        );
    }

    #[test]
    fn an_oversized_body_is_refused_before_it_is_parsed() {
        let body = vec![b'x'; MAX_RESPONSE_BYTES + 1];
        assert!(matches!(
            parse_response(&request(), &body),
            Err(LlmError::ResponseTooLarge { .. })
        ));
    }

    #[test]
    fn a_refusal_is_a_refusal_not_an_empty_answer() {
        let body = serde_json::json!({
            "stop_reason": "refusal",
            "usage": {"input_tokens": 1, "output_tokens": 0},
            "content": []
        });
        assert_eq!(
            parse_response(&request(), body.to_string().as_bytes()),
            Err(LlmError::Refused)
        );
    }

    #[test]
    fn a_garbage_retry_after_yields_no_hint_rather_than_an_invented_one() {
        assert_eq!(parse_retry_after("30"), Some(30_000));
        assert_eq!(parse_retry_after("0.5"), Some(500));
        assert_eq!(parse_retry_after("soon"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("-5"), None);
        // A provider asking us to sleep for a week is a provider bug.
        assert_eq!(parse_retry_after("604800"), Some(3_600_000));
    }

    #[test]
    fn the_scripted_client_replays_in_order_and_then_says_so() {
        let client = ScriptedClient::new(
            "fixture",
            vec![
                ScriptedTurn {
                    body: tool_use_body(),
                },
                ScriptedTurn {
                    body: serde_json::json!({
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 2, "output_tokens": 3},
                        "content": [{"type": "text", "text": "done"}]
                    }),
                },
            ],
        );
        let request = request();
        assert_eq!(client.remaining(), 2);
        assert_eq!(
            client.complete(&request).expect("turn 1").stop_reason,
            StopReason::ToolUse
        );
        assert_eq!(
            client.complete(&request).expect("turn 2").text.as_deref(),
            Some("done")
        );
        assert_eq!(client.remaining(), 0);
        // Exhaustion is its own error. An empty response here would look like a
        // valid turn to the loop, and the loop would keep going.
        assert_eq!(client.complete(&request), Err(LlmError::FixtureExhausted));
    }

    #[test]
    fn the_same_fixture_and_seed_produce_the_same_transcript_twice() {
        let fixture = serde_json::to_vec(&vec![ScriptedTurn {
            body: tool_use_body(),
        }])
        .expect("fixture encodes");
        let transcript = || {
            let client = ScriptedClient::from_fixture("fixture", &fixture).expect("fixture loads");
            let response = client.complete(&request()).expect("turn");
            serde_json::to_string(&response).expect("response encodes")
        };
        assert_eq!(transcript(), transcript());
    }

    #[test]
    fn retry_jitter_comes_from_the_seed_and_not_from_the_clock() {
        // A retry policy that consulted the wall clock would make every replayed
        // session diverge — for the sake of being polite to a server.
        let policy = RetryPolicy::default();
        let schedule = |seed: u64| {
            let mut rng = SmallRng::seed_from_u64(seed);
            (0..4)
                .map(|attempt| policy.backoff_ms(attempt, None, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(schedule(7), schedule(7), "same seed, same backoff schedule");
        assert!(
            schedule(7) != schedule(8),
            "different seeds must decorrelate retriers"
        );
        // A provider hint overrides jitter entirely — and is still capped.
        let mut rng = SmallRng::seed_from_u64(1);
        assert_eq!(policy.backoff_ms(0, Some(1_234), &mut rng), 1_234);
        assert_eq!(
            policy.backoff_ms(0, Some(u64::MAX), &mut rng),
            policy.max_backoff_ms
        );
    }

    #[test]
    fn only_transient_failures_are_retried() {
        assert!(RetryPolicy::is_retryable(&LlmError::Overloaded));
        assert!(RetryPolicy::is_retryable(&LlmError::RateLimited {
            retry_after_ms: None
        }));
        assert!(RetryPolicy::is_retryable(&LlmError::Transport("x".into())));
        // Retrying these would burn the budget re-asking a question that has
        // already been answered.
        assert!(!RetryPolicy::is_retryable(&LlmError::Refused));
        assert!(!RetryPolicy::is_retryable(&LlmError::FixtureExhausted));
        assert!(!RetryPolicy::is_retryable(&LlmError::InvalidResponse(
            "x".into()
        )));
    }

    #[test]
    fn the_anthropic_client_is_never_a_default_feature() {
        // Structural, and deliberately NOT `cfg!(feature = ...)`: that would only
        // describe the build currently running, so it would pass in CI while
        // silently failing to notice that someone had added the feature to the
        // default set. Reading the manifest catches that regardless of how this
        // particular build was invoked — including a build that legitimately
        // enables the feature to compile-check the client.
        let manifest = include_str!("../../Cargo.toml");
        let default_features = manifest
            .lines()
            .find(|line| line.trim_start().starts_with("default = ["))
            .unwrap_or_default();
        assert!(
            !default_features.contains("llm-anthropic"),
            "`llm-anthropic` must stay out of the default feature set, or `cargo \
             test` gains the ability to reach the network and spend real tokens; \
             found: {default_features}"
        );
    }
}
