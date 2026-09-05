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
/// The only model tool allowed to propose work to the autonomous lab.
pub const PROPOSE_EXPERIMENT_TOOL_NAME: &str = "propose_experiment";

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
    /// Provider error (HTTP 4xx / 5xx) with structured metadata.
    #[error("provider error {status}: {message}")]
    ProviderError {
        /// HTTP status code returned.
        status: u16,
        /// Provider error message, if parsed.
        message: String,
        /// Provider request ID, if supplied.
        request_id: Option<String>,
        /// Partial token usage, if supplied.
        partial_usage: Option<TokenUsage>,
    },
}

impl LlmError {
    /// Provider request identifier, if supplied with this failure.
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::ProviderError { request_id, .. } => request_id.as_deref(),
            _ => None,
        }
    }

    /// Partial token usage reported with this failure, if supplied.
    #[must_use]
    pub fn partial_usage(&self) -> Option<TokenUsage> {
        match self {
            Self::ProviderError { partial_usage, .. } => *partial_usage,
            _ => None,
        }
    }
}

/// Build the proposal tool from the single canonical experiment contract.
///
/// # Errors
///
/// [`LlmError::InvalidResponse`] if the generated schema cannot be represented
/// as JSON. This is a local contract failure, surfaced before any provider call.
pub fn propose_experiment_tool() -> Result<LlmToolSpec, LlmError> {
    let input_schema = crate::lab::spec::tool_input_schema().map_err(|error| {
        LlmError::InvalidResponse(format!(
            "canonical experiment tool schema is not encodable: {error}"
        ))
    })?;
    Ok(LlmToolSpec {
        name: PROPOSE_EXPERIMENT_TOOL_NAME.to_owned(),
        description: "Propose one bounded, falsifiable matched-seed experiment. Every factor, \
                      seed, metric, arm, and budget is validated before a run can start."
            .to_owned(),
        input_schema,
    })
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
    /// Provider request identifier, if returned in headers or body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
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

/// Read at most `cap + 1` bytes from a reader into a `Vec<u8>` without buffering an unbounded stream (bd-wqnk).
///
/// # Errors
///
/// Returns [`LlmError::ResponseTooLarge`] if `reader` yields more than `cap` bytes,
/// or [`LlmError::Transport`] if an I/O error occurs.
pub fn read_bounded_bytes<R: std::io::Read>(reader: R, cap: usize) -> Result<Vec<u8>, LlmError> {
    use std::io::Read;
    let mut bytes = Vec::new();
    let mut limited = reader.take((cap + 1) as u64);
    limited
        .read_to_end(&mut bytes)
        .map_err(|err| LlmError::Transport(err.to_string()))?;
    if bytes.len() > cap {
        return Err(LlmError::ResponseTooLarge {
            bytes: bytes.len(),
            cap,
        });
    }
    Ok(bytes)
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

    let request_id = value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);

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
        request_id,
    })
}

/// Parse an Anthropic error response body into structured error details.
#[must_use]
pub fn parse_error_body(status: u16, body: &[u8], request_id: Option<String>) -> LlmError {
    if body.is_empty() {
        return LlmError::ProviderError {
            status,
            message: format!("provider returned HTTP {status} with empty body"),
            request_id,
            partial_usage: None,
        };
    }
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
        let message = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("provider returned HTTP {status}"));

        let body_req_id = value
            .get("request_id")
            .or_else(|| value.get("id"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);

        let partial_usage = value.get("usage").and_then(|u| {
            let input = u.get("input_tokens").and_then(serde_json::Value::as_u64)?;
            let output = u.get("output_tokens").and_then(serde_json::Value::as_u64)?;
            Some(TokenUsage {
                input: u32::try_from(input).ok()?,
                output: u32::try_from(output).ok()?,
            })
        });

        LlmError::ProviderError {
            status,
            message,
            request_id: request_id.or(body_req_id),
            partial_usage,
        }
    } else {
        let truncated_len = body.len().min(256);
        let preview = body.get(..truncated_len).map_or_else(
            || String::from_utf8_lossy(body).into_owned(),
            |slice| String::from_utf8_lossy(slice).into_owned(),
        );
        LlmError::ProviderError {
            status,
            message: format!("HTTP {status}: {preview}"),
            request_id,
            partial_usage: None,
        }
    }
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
        match error {
            LlmError::Transport(_) | LlmError::RateLimited { .. } | LlmError::Overloaded => true,
            LlmError::ProviderError { status, .. } => *status >= 500 && *status != 529,
            _ => false,
        }
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

/// Belt and braces: strip the key from any string that is about to be
/// surfaced in an error, a log line, or a notebook (bd-wqnk).
#[allow(dead_code)]
pub(crate) fn redact(message: &str, api_key: &str) -> String {
    if api_key.is_empty() {
        return message.to_owned();
    }
    message.replace(api_key, "<redacted>")
}

#[cfg(feature = "llm-anthropic")]
pub use anthropic::AnthropicClient;

#[cfg(feature = "llm-anthropic")]
mod anthropic {
    use super::{
        LlmClient, LlmError, LlmRequest, LlmResponse, MAX_RESPONSE_BYTES, RetryPolicy,
        parse_error_body, parse_response, parse_retry_after, read_bounded_bytes, redact,
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
            let start = std::time::Instant::now();
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
            let declared_bytes = response
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|val| val.to_str().ok())
                .and_then(|s| s.parse::<usize>().ok());

            let request_id = response
                .headers()
                .get("request-id")
                .or_else(|| response.headers().get("x-request-id"))
                .and_then(|val| val.to_str().ok())
                .map(ToOwned::to_owned);

            if status.as_u16() == 429 {
                let retry_after_ms = response
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .and_then(parse_retry_after);
                tracing::warn!(
                    provider = "anthropic",
                    status = 429,
                    retry_after_ms = ?retry_after_ms,
                    retry_classification = "retryable",
                    latency_ms = start.elapsed().as_millis() as u64,
                    request_id = ?request_id,
                    "rate limited by provider"
                );
                return Err(LlmError::RateLimited { retry_after_ms });
            }
            if status.as_u16() == 529 {
                tracing::warn!(
                    provider = "anthropic",
                    status = 529,
                    retry_classification = "retryable",
                    latency_ms = start.elapsed().as_millis() as u64,
                    request_id = ?request_id,
                    "provider overloaded"
                );
                return Err(LlmError::Overloaded);
            }

            // Read incrementally into a bounded buffer: past the cap we stop, rather than
            // letting a pathological body grow until the process dies (bd-wqnk).
            let read_res = read_bounded_bytes(response, MAX_RESPONSE_BYTES);
            let bytes = match read_res {
                Ok(b) => b,
                Err(err) => {
                    let observed_bytes = match err {
                        LlmError::ResponseTooLarge { bytes, .. } => bytes,
                        _ => 0,
                    };
                    tracing::warn!(
                        provider = "anthropic",
                        status = status.as_u16(),
                        declared_bytes = ?declared_bytes,
                        observed_bytes,
                        cap_decision = "rejected",
                        latency_ms = start.elapsed().as_millis() as u64,
                        retry_classification = "terminal",
                        request_id = ?request_id,
                        "response body rejected by byte cap or transport error"
                    );
                    return Err(match err {
                        LlmError::Transport(msg) => {
                            LlmError::Transport(redact(&msg, &self.api_key))
                        }
                        other => other,
                    });
                }
            };

            let observed_bytes = bytes.len();
            if !status.is_success() {
                let err = parse_error_body(status.as_u16(), &bytes, request_id);
                let retry_class = if RetryPolicy::is_retryable(&err) {
                    "retryable"
                } else {
                    "terminal"
                };
                tracing::warn!(
                    provider = "anthropic",
                    status = status.as_u16(),
                    declared_bytes = ?declared_bytes,
                    observed_bytes,
                    cap_decision = "accepted",
                    latency_ms = start.elapsed().as_millis() as u64,
                    retry_classification = retry_class,
                    request_id = ?err.request_id(),
                    partial_input_tokens = err.partial_usage().map(|u| u.input),
                    partial_output_tokens = err.partial_usage().map(|u| u.output),
                    "provider error response received"
                );
                return Err(err);
            }

            let mut parsed = parse_response(request, &bytes)?;
            if parsed.request_id.is_none() {
                parsed.request_id = request_id;
            }

            tracing::info!(
                provider = "anthropic",
                model_id = %self.model_id,
                status = status.as_u16(),
                declared_bytes = ?declared_bytes,
                observed_bytes,
                cap_decision = "accepted",
                latency_ms = start.elapsed().as_millis() as u64,
                input_tokens = parsed.usage.input,
                output_tokens = parsed.usage.output,
                stop_reason = ?parsed.stop_reason,
                request_id = ?parsed.request_id,
                "llm call attempt succeeded"
            );

            Ok(parsed)
        }
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
                            request_id = ?response.request_id,
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
                            retry_classification = "retryable",
                            request_id = ?error.request_id(),
                            partial_usage = ?error.partial_usage(),
                            "llm call failed; retrying"
                        );
                        last = error;
                        if attempt + 1 < self.policy.attempts {
                            std::thread::sleep(std::time::Duration::from_millis(backoff));
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            attempt,
                            error_kind = ?error,
                            retry_classification = "terminal",
                            request_id = ?error.request_id(),
                            partial_usage = ?error.partial_usage(),
                            "llm call failed terminally"
                        );
                        return Err(error);
                    }
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
            tools: vec![propose_experiment_tool().expect("canonical proposal tool")],
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
    fn proposal_tool_schema_is_exactly_the_canonical_schema() {
        let tool = propose_experiment_tool().expect("canonical proposal tool");
        assert_eq!(tool.name, PROPOSE_EXPERIMENT_TOOL_NAME);
        assert_eq!(
            tool.input_schema,
            crate::lab::spec::tool_input_schema().expect("canonical schema")
        );
        assert!(tool.input_schema.pointer("/properties/factors").is_some());
        assert!(
            tool.input_schema
                .pointer("/x-scriptbots-knob-ranges")
                .is_some()
        );
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

    #[test]
    fn read_bounded_bytes_enforces_cap_without_buffering_unbounded_streams() {
        use std::io::Cursor;

        // Cap - 1: exact read succeeds
        let data = vec![b'a'; 10];
        let res = read_bounded_bytes(Cursor::new(data.clone()), 11).expect("cap - 1 succeeds");
        assert_eq!(res, data);

        // Cap exact: exact read succeeds
        let res = read_bounded_bytes(Cursor::new(data.clone()), 10).expect("cap exact succeeds");
        assert_eq!(res, data);

        // Cap + 1: fails with ResponseTooLarge and buffers at most cap + 1
        let res = read_bounded_bytes(Cursor::new(vec![b'b'; 11]), 10);
        assert_eq!(res, Err(LlmError::ResponseTooLarge { bytes: 11, cap: 10 }));

        // Infinite stream: stops reading at cap + 1 and does not OOM
        let infinite = std::io::repeat(b'z');
        let res = read_bounded_bytes(infinite, 100);
        assert_eq!(
            res,
            Err(LlmError::ResponseTooLarge {
                bytes: 101,
                cap: 100,
            })
        );
    }

    #[test]
    fn api_key_redaction_replaces_secrets_and_handles_empty_keys() {
        let key = "sk-ant-api03-secretkey12345";
        let err_msg = format!("request to https://api.anthropic.com failed with key {key}");
        let redacted = super::redact(&err_msg, key);
        assert!(!redacted.contains(key));
        assert!(redacted.contains("<redacted>"));

        let empty_redacted = super::redact("some error", "");
        assert_eq!(empty_redacted, "some error");
    }

    struct ChunkedReader<'a> {
        data: &'a [u8],
        chunk_size: usize,
        pos: usize,
    }

    impl std::io::Read for ChunkedReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let remaining = self.data.len().saturating_sub(self.pos);
            let to_read = remaining.min(self.chunk_size).min(buf.len());
            if let (Some(src), Some(dst)) = (
                self.data.get(self.pos..self.pos + to_read),
                buf.get_mut(..to_read),
            ) {
                dst.copy_from_slice(src);
            }
            self.pos += to_read;
            Ok(to_read)
        }
    }

    #[test]
    fn read_bounded_bytes_handles_adversarial_chunk_sizes() {
        let payload = vec![b'x'; 500];
        // 1-byte chunking under cap
        let mut reader = ChunkedReader {
            data: &payload,
            chunk_size: 1,
            pos: 0,
        };
        let res = read_bounded_bytes(&mut reader, 500).expect("exact cap succeeds");
        assert_eq!(res.len(), 500);

        // 1-byte chunking over cap
        let mut reader = ChunkedReader {
            data: &payload,
            chunk_size: 1,
            pos: 0,
        };
        let res = read_bounded_bytes(&mut reader, 499);
        assert_eq!(
            res,
            Err(LlmError::ResponseTooLarge {
                bytes: 500,
                cap: 499,
            })
        );

        // 7-byte chunking over cap
        let mut reader = ChunkedReader {
            data: &payload,
            chunk_size: 7,
            pos: 0,
        };
        let res = read_bounded_bytes(&mut reader, 100);
        assert_eq!(
            res,
            Err(LlmError::ResponseTooLarge {
                bytes: 101,
                cap: 100,
            })
        );
    }

    struct FailingReader {
        yielded: usize,
        fail_at: usize,
    }

    impl std::io::Read for FailingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.yielded >= self.fail_at {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "connection reset by peer",
                ));
            }
            let to_write = self.fail_at.saturating_sub(self.yielded).min(buf.len());
            if let Some(dst) = buf.get_mut(..to_write) {
                dst.fill(b'q');
            }
            self.yielded += to_write;
            Ok(to_write)
        }
    }

    #[test]
    fn read_bounded_bytes_returns_transport_on_connection_abort() {
        let mut reader = FailingReader {
            yielded: 0,
            fail_at: 50,
        };
        let res = read_bounded_bytes(&mut reader, 100);
        assert!(
            matches!(res, Err(LlmError::Transport(msg)) if msg.contains("connection reset by peer"))
        );
    }

    #[test]
    fn parse_error_body_extracts_details_and_partial_usage() {
        let err_json = serde_json::json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Prompt exceeded token window"
            },
            "request_id": "req_019abc",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 0
            }
        });
        let err = parse_error_body(400, err_json.to_string().as_bytes(), None);
        assert_eq!(err.request_id(), Some("req_019abc"));
        assert_eq!(
            err.partial_usage(),
            Some(TokenUsage {
                input: 120,
                output: 0,
            })
        );
        assert_eq!(
            err,
            LlmError::ProviderError {
                status: 400,
                message: "Prompt exceeded token window".to_owned(),
                request_id: Some("req_019abc".to_owned()),
                partial_usage: Some(TokenUsage {
                    input: 120,
                    output: 0,
                }),
            }
        );
        assert!(!RetryPolicy::is_retryable(&err));

        // 500 server error is retryable
        let server_err =
            parse_error_body(500, b"Internal Server Error", Some("req_500".to_owned()));
        assert_eq!(server_err.request_id(), Some("req_500"));
        assert!(RetryPolicy::is_retryable(&server_err));
    }

    #[test]
    fn tracing_capture_verifies_redacted_diagnostics() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::fmt::MakeWriter;

        #[derive(Clone)]
        struct SharedLog(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for SharedLog {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().expect("log lock").extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> MakeWriter<'a> for SharedLog {
            type Writer = SharedLog;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedLog(buffer.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_ansi(false)
            .finish();

        let sentinel = format!("{}-{}", "sk-ant-api03", "TEST-SENTINEL-12345");
        let raw_error = format!("failed request with secret {sentinel} at endpoint");
        let safe_error = super::redact(&raw_error, &sentinel);

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                provider = "anthropic",
                model_id = "claude-3-5-sonnet",
                status = 200,
                declared_bytes = 100,
                observed_bytes = 100,
                cap_decision = "accepted",
                latency_ms = 45,
                retry_classification = "none",
                "simulated success"
            );
            tracing::warn!(
                provider = "anthropic",
                error = %safe_error,
                "simulated error warning"
            );
        });

        let output =
            String::from_utf8(buffer.lock().expect("read log buffer").clone()).expect("utf8 log");
        assert!(
            !output.contains(&sentinel),
            "sentinel secret leaked into tracing logs: {output}"
        );
        assert!(output.contains("<redacted>"));
        assert!(output.contains("cap_decision=\"accepted\""));
    }

    #[test]
    #[ignore = "requires ANTHROPIC_API_KEY; live network roundtrip"]
    fn live_anthropic_provider_roundtrip_records_redacted_diagnostics() {
        let auth_env = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        if auth_env.is_empty() {
            eprintln!("skipping live provider test: ANTHROPIC_API_KEY is not set");
        }
        #[cfg(feature = "llm-anthropic")]
        if !auth_env.is_empty() {
            let rng = SmallRng::seed_from_u64(0x1234_5678);
            let client = AnthropicClient::from_env("claude-3-5-haiku-20241022", rng)
                .expect("client builds from environment");
            let req = LlmRequest {
                system: "respond in one short sentence".to_owned(),
                messages: vec![LlmMessage {
                    role: "user".to_owned(),
                    content: "say hello".to_owned(),
                }],
                tools: Vec::new(),
                max_tokens: 32,
            };
            let resp = client.complete(&req).expect("live request succeeds");
            assert!(resp.text.is_some());
            assert!(resp.usage.input > 0);
            assert!(resp.usage.output > 0);
            eprintln!(
                "live Anthropic roundtrip succeeded: request_id={:?}",
                resp.request_id
            );
        }
    }
}
