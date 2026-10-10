use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use scriptbots_app::{
    ControlCommand, ControlRuntime, ControlServerConfig, McpTransportConfig, ScenarioIdentityV0,
    host_thread::HostThread,
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
use scriptbots_core::{
    AgentData, Generation, PersistenceAdmissionSession, Position, ScriptBotsConfig, Velocity,
    WorldState,
};
use scriptbots_runtime::{
    HostCoreOptions, HostSessionId, JournalPort, PlaybackSnapshot, VolatileJournal,
    channel::ChannelHostOptions,
};
use scriptbots_storage::{AnalyticsSnapshotProvider, StoragePipeline, StorageReader};
use serde::Deserialize;
use serial_test::serial;
use tempfile::tempdir;

static ENV_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

/// Exercise the shipped interactive entry point with a controlling terminal.
/// Keep raw output and termios observations under the DSR proof's scratch root.
#[cfg(target_os = "linux")]
#[test]
fn live_frankentui_pty_routes_and_restores_terminal() -> Result<()> {
    // ubs:ignore[rust.security.shell-command] — fixed Python probe; variable input is separate argv.
    let output = std::process::Command::new("python3")
        .args(["-c", LIVE_TERMINAL_PTY_PROBE])
        .arg(env!("CARGO_BIN_EXE_scriptbots-app"))
        .arg(option_env!("SCRIPTBOTS_SOURCE_REVISION").unwrap_or("unavailable"))
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("{stdout}");
    assert!(
        output.status.success(),
        "actual PTY probe failed: {}\n{stdout}\n{stderr}",
        output.status
    );
    let observations: Vec<serde_json::Value> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert_eq!(
        observations.len(),
        2,
        "both capability profiles must execute"
    );
    for observation in observations {
        assert_eq!(observation["schema"], "scriptbots.live-tui-pty.v1");
        assert_eq!(observation["exit_code"], 0);
        assert_eq!(observation["raw_mode_observed"], true);
        assert_eq!(observation["terminal_attributes_restored"], true);
        assert_eq!(observation["alternate_screen_restored"], true);
        assert_eq!(observation["cursor_hidden_observed"], true);
        assert_eq!(observation["cursor_visible_at_exit"], true);
        assert_eq!(
            observation["enabled_capture_modes_at_exit"],
            serde_json::json!([])
        );
        assert_eq!(observation["frankentui_shutdown_observed"], true);
        assert_eq!(observation["binary_source_observed"], true);
        let interaction = &observation["byte_input_resize_evidence"];
        assert_eq!(interaction["source_commit"], observation["source_commit"]);
        assert_eq!(interaction["owner_tick_unchanged"], true);
        assert_eq!(interaction["responsiveness_observed"], true);
        assert_eq!(
            interaction["observed_resize_sequence"],
            interaction["requested_resize_sequence"]
        );
        let resize_probes = interaction["observed_resize_probes"]
            .as_array()
            .expect("actual resize observations must be present");
        assert!(!resize_probes.is_empty());
        for probe in resize_probes {
            assert_eq!(
                probe["footer_geometry"]["terminal"],
                probe["footer_geometry"]["presented"]
            );
            assert!(probe["footer_geometry"]["terminal"].is_object());
            assert_eq!(
                probe["stale_cell_control"]["requested_dimensions_observed"],
                true
            );
            assert_eq!(probe["stale_cell_control"]["full_heading_preserved"], true);
            assert_eq!(
                probe["stale_cell_control"]["parser_sequence_accepted"],
                true
            );
            assert_eq!(probe["stale_cell_control"]["rejected"], true);
            assert!(probe["stale_cell_control"]["observed_footer_geometry"].is_null());
        }
        assert_eq!(
            interaction["observed_arrow_probes"]
                .as_array()
                .map(|probes| {
                    probes
                        .iter()
                        .map(|probe| probe["key"].as_str())
                        .collect::<Vec<_>>()
                }),
            Some(vec![Some("down"), Some("up")])
        );
        assert_eq!(
            interaction["mouse_navigation"]["observed_route"],
            "Lineages"
        );
        assert_eq!(
            interaction["mouse_navigation"]["observed_sgr_capture"],
            true
        );
        assert!(
            observation["python_identity"]["executable_sha256"]
                .as_str()
                .is_some_and(|hash| hash.len() == 64)
        );
        assert!(
            observation["parser_guard_evidence"]["rejected_cases"]
                .as_array()
                .is_some_and(|cases| !cases.is_empty())
        );
        assert!(
            observation["parser_guard_evidence"]["valid_stream_byte_length"]
                .as_u64()
                .is_some_and(|length| length > 0)
        );
        assert_eq!(
            observation["source_commit"],
            option_env!("SCRIPTBOTS_SOURCE_REVISION").unwrap_or("unavailable")
        );
        match observation["capability_profile"].as_str() {
            Some("truecolor") => {
                assert!(
                    observation["observed_truecolor_sgr_count"]
                        .as_u64()
                        .is_some_and(|count| count > 0),
                    "truecolor must be observed in actual terminal output"
                );
                let evidence = &observation["data_provider_evidence"];
                assert_eq!(evidence["checkpoint"]["new_catalog_entries"], 1);
                assert_eq!(evidence["experiment"]["status"], "completed");
                assert_eq!(evidence["experiment"]["completed_runs"], 1);
                assert_eq!(evidence["experiment"]["failed_runs"], 0);
                assert_eq!(evidence["refresh"]["terminal_input_count_during_change"], 0);
                assert_eq!(
                    evidence["refresh"]["pause_receipt"]["application_state"],
                    "applied"
                );
                assert_eq!(
                    evidence["refresh"]["step_receipt"]["application_state"],
                    "applied"
                );
                assert!(evidence["refresh"]["before_tick"].is_u64());
                assert_eq!(
                    evidence["refresh"]["after_tick"].as_u64(),
                    evidence["refresh"]["before_tick"]
                        .as_u64()
                        .and_then(|tick| tick.checked_add(1))
                );
            }
            Some("no-color") => assert_eq!(observation["observed_color_sgr_count"], 0),
            // ubs:ignore[rust.ownership.panic-macro] — test requires a declared capability label.
            profile => panic!("unexpected capability profile: {profile:?}"),
        }
        assert_eq!(
            observation["observed_screenshot_routes"]
                .as_array()
                .map(Vec::len),
            Some(6)
        );
        assert_eq!(
            observation["observed_science_routes"]
                .as_array()
                .map(Vec::len),
            Some(6)
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
const LIVE_TERMINAL_PTY_PROBE: &str = r#"
import codecs, fcntl, hashlib, json, os, pathlib, pty, re, select, signal
import socket, struct, subprocess, sys, tempfile, termios, time, urllib.error, urllib.request
import unicodedata

binary, source = sys.argv[1:]
routes = [
    ('4', 'LINEAGES & PHYLOGENY LAB'),
    ('5', 'BRAIN ARENA & EVALUATOR COHORTS'),
    ('6', 'MATCHED-SEED EXPERIMENTS & INTERVENTIONS'),
    ('7', 'DETERMINISTIC REPLAY & FIRST-DIVERGENCE TRACKER'),
    ('8', 'ENVIRONMENT, HYDROLOGY & BIOME LAYERS'),
    ('9', 'SYSTEM DIAGNOSTICS & PIPELINE HEALTH'),
]
class TerminalScreen:
    # The pinned presenter emits differential cursor-addressed updates. Keep the
    # actual visible cells instead of treating its output as an append-only log.
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cells = [[' '] * width for _ in range(height)]
        self.x = self.y = 0
        self.saved = (0, 0)
        self.scroll_top, self.scroll_bottom = 0, height - 1
        self.wrap_pending = False
        self.autowrap = True
        self.alternate = False
        self.alternate_entered = self.alternate_left = False
        self.synchronized = False
        self.cursor_visible = True
        self.cursor_hidden_observed = self.cursor_show_observed = False
        self.enabled_capture_modes = set()
        self.color_count = self.truecolor_count = self.attribute_count = 0
        self.pending = ''
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')

    def text(self):
        return '\n'.join(''.join(row) for row in self.cells)

    def resize(self, width, height):
        assert 1 <= width <= 300 and 1 <= height <= 80, 'bounded terminal dimensions exceeded'
        old_width, old_height = self.width, self.height
        full_scroll_region = self.scroll_top == 0 and self.scroll_bottom == old_height - 1
        cells = [row[:width] + [' '] * max(0, width - old_width)
                 for row in self.cells[:height]]
        cells.extend([[' '] * width for _ in range(max(0, height - old_height))])
        for row in cells:
            if row[-1] and unicodedata.east_asian_width(row[-1][0]) in ('W', 'F'):
                row[-1] = ' '  # A clipped wide glyph cannot retain a missing continuation.
        self.width, self.height, self.cells = width, height, cells
        self.x, self.y = min(self.x, width - 1), min(self.y, height - 1)
        self.saved = (min(self.saved[0], width - 1), min(self.saved[1], height - 1))
        self.wrap_pending = self.wrap_pending and width == old_width
        if full_scroll_region or self.scroll_top >= height - 1:
            self.scroll_top, self.scroll_bottom = 0, height - 1
        else:
            self.scroll_bottom = min(self.scroll_bottom, height - 1)
        # Deliberately preserve decoder and pending: resizing cannot repair a
        # torn UTF-8 character or CSI already received from the actual child.

    def position(self, x, y):
        self.x = max(0, min(self.width - 1, x))
        self.y = max(0, min(self.height - 1, y))
        self.wrap_pending = False

    def line_feed(self):
        self.wrap_pending = False
        if self.y == self.scroll_bottom:
            self.cells[self.scroll_top:self.scroll_bottom + 1] = (
                self.cells[self.scroll_top + 1:self.scroll_bottom + 1]
                + [[' '] * self.width])
        else:
            self.y = min(self.height - 1, self.y + 1)

    def put(self, character):
        if unicodedata.category(character) in ('Mn', 'Me'):
            if self.x > 0 or self.wrap_pending:
                column = self.x if self.wrap_pending else self.x - 1
                self.cells[self.y][column] += character
            return
        if unicodedata.category(character) == 'Cf':
            raise AssertionError('unsupported terminal format character: ' + repr(character))
        width = 2 if unicodedata.east_asian_width(character) in ('W', 'F') else 1
        if self.autowrap and (self.wrap_pending or self.x + width > self.width):
            self.line_feed()
            self.x = 0
        self.cells[self.y][self.x] = character
        if width == 2 and self.x + 1 < self.width:
            self.cells[self.y][self.x + 1] = ''
        following = self.x + width
        self.wrap_pending = following >= self.width
        self.x = min(self.width - 1, following)

    def sgr(self, values):
        index = 0
        while index < len(values):
            code = values[index]
            if code in (38, 48):
                assert index + 1 < len(values), 'incomplete extended SGR color'
                mode = values[index + 1]
                count = 3 if mode == 2 else 1 if mode == 5 else 0
                assert count and index + 2 + count <= len(values), 'unsupported SGR color'
                assert all(0 <= value <= 255 for value in values[index + 2:index + 2 + count])
                self.color_count += 1
                self.truecolor_count += int(mode == 2)
                index += count + 2
                continue
            if 30 <= code <= 37 or 40 <= code <= 47 or 90 <= code <= 97 or 100 <= code <= 107:
                self.color_count += 1
            elif code in (1, 2, 3, 4, 5, 7, 8, 9, 22, 23, 24, 25, 27, 28, 29):
                self.attribute_count += 1
            elif code not in (0, 39, 49):
                raise AssertionError('unsupported SGR attribute: ' + str(code))
            index += 1

    def csi(self, parameters, final):
        if parameters.startswith('?') and final in ('h', 'l'):
            modes = [int(value) for value in parameters[1:].split(';')]
            allowed = {7, 25, 1000, 1001, 1002, 1003, 1004, 1005, 1006,
                       1015, 1016, 1049, 2004, 2026}
            assert set(modes) <= allowed, 'unsupported terminal mode: ' + parameters + final
            for mode in modes:
                enabled = final == 'h'
                if mode not in (7, 25):
                    if enabled:
                        self.enabled_capture_modes.add(mode)
                    else:
                        self.enabled_capture_modes.discard(mode)
                if mode == 7:
                    self.autowrap = enabled
                elif mode == 25:
                    self.cursor_visible = enabled
                    self.cursor_hidden_observed |= not enabled
                    self.cursor_show_observed |= enabled
                elif mode == 1049:
                    self.alternate = enabled
                    if enabled:
                        self.alternate_entered = True
                        self.cells = [[' '] * self.width for _ in range(self.height)]
                        self.position(0, 0)
                    else:
                        self.alternate_left = True
                elif mode == 2026:
                    self.synchronized = enabled
            return
        if final == 'u' and parameters in ('>15', '<'):
            return  # Pinned session's Kitty keyboard push/pop; no screen effect.
        assert re.fullmatch(r'[0-9;]*', parameters), 'unsupported CSI: ' + parameters + final
        values = [int(value or 0) for value in parameters.split(';')]
        amount = values[0] or 1
        if final in ('H', 'f') and len(values) <= 2:
            self.position((values[1] or 1) - 1 if len(values) == 2 else 0, amount - 1)
        elif final == 'G' and len(values) == 1:
            self.position(amount - 1, self.y)
        elif final in ('A', 'B', 'C', 'D') and len(values) == 1:
            self.position(self.x + (amount if final == 'C' else -amount if final == 'D' else 0),
                          self.y + (amount if final == 'B' else -amount if final == 'A' else 0))
        elif final == 'm':
            self.sgr(values)
        elif final in ('J', 'K') and len(values) == 1:
            mode = values[0]
            assert mode in (0, 1, 2) or (final == 'J' and mode == 3), 'unsupported erase mode'
            if mode == 3:
                return  # Erases scrollback; the bounded visible screen is unchanged.
            rows = range(self.height) if final == 'J' else [self.y]
            for row in rows:
                for column in range(self.width):
                    current = (row, column)
                    cursor = (self.y, self.x)
                    if mode == 2 or (mode == 0 and current >= cursor) or (mode == 1 and current <= cursor):
                        self.cells[row][column] = ' '
        elif final == 'r' and len(values) <= 2:
            top = (values[0] or 1) - 1
            bottom = (values[1] or self.height) - 1 if len(values) == 2 else self.height - 1
            assert 0 <= top < bottom < self.height, 'unsupported scroll margins'
            self.scroll_top, self.scroll_bottom = top, bottom
            self.position(0, 0)
        else:
            raise AssertionError('unsupported CSI: ' + repr(parameters + final))

    def feed(self, chunk, final=False):
        data = self.pending + self.decoder.decode(chunk, final=final)
        self.pending = ''
        index = 0
        while index < len(data):
            character = data[index]
            if character == '\x1b':
                start = index
                if index + 1 == len(data):
                    break
                kind = data[index + 1]
                if kind == '[':
                    end = index + 2
                    while end < len(data) and not ('@' <= data[end] <= '~'):
                        assert ' ' <= data[end] <= '?', 'invalid CSI byte'
                        assert end - start < 256, 'bounded CSI exceeded'
                        end += 1
                    if end == len(data):
                        break
                    self.csi(data[index + 2:end], data[end])
                    index = end + 1
                    continue
                if kind == ']':
                    end = index + 2
                    while end < len(data) and data[end] != '\x07' and data[end:end + 2] != '\x1b\\':
                        assert end - start < 8192, 'bounded OSC exceeded'
                        end += 1
                    if end == len(data) or (data[end] == '\x1b' and end + 1 == len(data)):
                        break
                    payload = data[index + 2:end]
                    assert payload.startswith('8;'), 'unsupported OSC: ' + repr(payload)
                    index = end + (1 if data[end] == '\x07' else 2)
                    continue
                if kind == '7':
                    self.saved = (self.x, self.y)
                elif kind == '8':
                    self.position(*self.saved)
                else:
                    raise AssertionError('unsupported escape: ' + repr(data[index:index + 2]))
                index += 2
                continue
            if character == '\r':
                self.position(0, self.y)
            elif character == '\n':
                self.line_feed()
            elif character == '\b':
                self.position(self.x - 1, self.y)
            elif character == '\t':
                self.position((self.x // 8 + 1) * 8, self.y)
            elif character == '\x07':
                pass
            elif ord(character) < 32 or ord(character) == 127:
                raise AssertionError('unsupported terminal control: ' + repr(character))
            else:
                self.put(character)
            index += 1
        self.pending = data[index:]
        assert len(self.pending) <= 8192, 'bounded partial terminal sequence exceeded'
        assert not final or not self.pending, 'incomplete terminal sequence at shutdown'

def diagnostics_footer_geometry(text, width, height):
    # Diagnostics rates and queue observations can vary while science is paused.
    # The shell draws its full-width footer divider after that body, followed by
    # exactly one final status row; toasts end above these two rows.
    rows = text.splitlines()
    if len(rows) != height:
        return None
    headings = [row for row, line in enumerate(rows) if routes[-1][1] in line]
    if len(headings) != 1:
        return None
    dividers = [row for row, line in enumerate(rows)
                if row > headings[0] and line == '-' * width]
    if len(dividers) != 1 or dividers[0] + 1 != len(rows) - 1:
        return None
    status_row = dividers[0] + 1
    if not rows[status_row].startswith(' ') or not rows[status_row].strip():
        return None
    return {'divider_row': dividers[0], 'divider_column': rows[dividers[0]].find('-'),
            'divider_width': len(rows[dividers[0]]), 'status_row': status_row,
            'status_text': rows[status_row]}

def check_parser_guards():
    # These are parser controls, not transcripts or substitute live evidence.
    accepted, rejected = [], []
    stream = (b'\x1b[?1049h\x1b[?25l\x1b[2J\x1b[2;2H' + 'Aé'.encode()
              + b'\x1b7\x1b[1;1HZ\x1b8\x1b[38;2;1;2;3mB\x1b[0m'
              + b'\x1b[?25h\x1b[?1049l')
    expected = 'Z       \n AéB    \n        '
    for split in range(len(stream) + 1):
        parser = TerminalScreen(8, 3)
        parser.feed(stream[:split])
        parser.feed(stream[split:], final=True)
        assert parser.text() == expected and (parser.x, parser.y) == (4, 1)
        assert parser.truecolor_count == parser.color_count == 1
        assert parser.cursor_hidden_observed and parser.cursor_show_observed and parser.cursor_visible
        assert parser.alternate_left and not parser.alternate and not parser.enabled_capture_modes
        accepted.append({'case': 'valid_stream_split', 'split_byte': split})
    parser = TerminalScreen(8, 3)
    for byte in stream:
        parser.feed(bytes([byte]))
    parser.feed(b'', final=True)
    assert parser.text() == expected
    accepted.append({'case': 'valid_stream_single_byte_chunks'})
    parser = TerminalScreen(16, 3)
    parser.feed(b'HELLO THERE\x1b[1;7HDAVE \x1b[2;1Hdiscard\x1b[2K', final=True)
    assert parser.text() == 'HELLO DAVE      \n                \n                '
    accepted.append({'case': 'differential_cursor_repaint_and_erase'})
    parser = TerminalScreen(8, 3)
    parser.feed(b'\x1b[3;8H\x1b7\x1b[2;')
    assert parser.pending == '\x1b[2;'
    parser.resize(5, 2)
    assert parser.pending == '\x1b[2;' and parser.saved == (4, 1)
    parser.feed(b'2HX\x1b8', final=True)
    assert parser.text() == '     \n X   ' and (parser.x, parser.y) == (4, 1)
    accepted.append({'case': 'resize_preserves_partial_csi_and_clamps_saved_cursor'})
    parser = TerminalScreen(8, 3)
    parser.feed(b'\xc3')
    parser.resize(5, 2)
    parser.feed(b'\xa9', final=True)
    assert parser.text() == 'é    \n     '
    parser.resize(8, 3)
    assert parser.text() == 'é       \n        \n        '
    accepted.append({'case': 'resize_preserves_partial_utf8_and_expands_blank_cells'})
    invalid = [
        ('invalid_utf8', b'\xff', UnicodeDecodeError, None),
        ('truncated_utf8', b'\xc3', UnicodeDecodeError, None),
        ('truncated_csi', b'\x1b[2;', AssertionError, 'incomplete terminal sequence'),
        ('truncated_osc', b'\x1b]8;;https://example.invalid', AssertionError, 'incomplete terminal sequence'),
        ('invalid_csi_byte', b'\x1b[2;\nH', AssertionError, 'invalid CSI byte'),
        ('incomplete_extended_color', b'\x1b[38;2;1;2m', AssertionError, 'unsupported SGR color'),
        ('unsupported_csi', b'\x1b[1z', AssertionError, 'unsupported CSI'),
        ('unsupported_escape', b'\x1bX', AssertionError, 'unsupported escape'),
        ('unsupported_mode', b'\x1b[?9999h', AssertionError, 'unsupported terminal mode'),
        ('unsupported_osc', b'\x1b]0;title\x07', AssertionError, 'unsupported OSC'),
        ('invalid_scroll_region', b'\x1b[3;2r', AssertionError, 'unsupported scroll margins'),
        ('oversized_csi', b'\x1b[' + b'1' * 300, AssertionError, 'bounded CSI exceeded'),
    ]
    for name, transcript, exception, message in invalid:
        parser = TerminalScreen(8, 3)
        try:
            parser.feed(transcript, final=True)
        except exception as error:
            assert message is None or message in str(error), (name, str(error))
            rejected.append({'case': name, 'exception': type(error).__name__, 'reason': str(error)})
        else:
            raise AssertionError('parser accepted negative control: ' + name)
    return {'synthetic_parser_controls': True, 'valid_stream_byte_length': len(stream),
            'accepted_cases': accepted, 'rejected_cases': rejected}

parser_guard_evidence = check_parser_guards()
python_identity = {'executable': os.path.realpath(sys.executable), 'version': sys.version,
                   'implementation': sys.implementation.name, 'cache_tag': sys.implementation.cache_tag,
                   'platform': sys.platform, 'unicode_version': unicodedata.unidata_version,
                   'stdlib_modules': {}}
python_identity['executable_sha256'] = hashlib.sha256(pathlib.Path(python_identity['executable']).read_bytes()).hexdigest()
for module in (pty, termios, fcntl, codecs, unicodedata):
    module_path = pathlib.Path(module.__file__).resolve()
    python_identity['stdlib_modules'][module.__name__] = {
        'path': str(module_path), 'sha256': hashlib.sha256(module_path.read_bytes()).hexdigest()}

for profile in ('truecolor', 'no-color'):
    directory = pathlib.Path(tempfile.mkdtemp(prefix='live-tui-' + profile + '-'))
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 36, 120, 0, 0))
    initial = termios.tcgetattr(slave)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('SCRIPTBOTS_', 'SB_'))}
    environment.update(SCRIPTBOTS_MODE='terminal', SCRIPTBOTS_TERMINAL_HEADLESS='0',
                       SCRIPTBOTS_TERMINAL_EMOJI='0',
                       SCRIPTBOTS_ARTIFACTS_DIR=str(directory / 'artifacts'),
                       SCRIPTBOTS_CONTROL_REST_ENABLED='1', SCRIPTBOTS_CONTROL_MCP='disabled',
                       SCRIPTBOTS_CONTROL_REST_ADDR='127.0.0.1:' + str(port),
                       SCRIPTBOTS_TERMINAL_LOG=str(directory / 'runtime.log'),
                       RUST_LOG='scriptbots::terminal=info', TERM='xterm-256color')
    if profile == 'no-color':
        environment['NO_COLOR'] = '1'
        environment.pop('COLORTERM', None)
    else:
        environment.pop('NO_COLOR', None)
        environment['COLORTERM'] = 'truecolor'
    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    raw = bytearray()
    screen = TerminalScreen(120, 36)
    observed = []
    screenshots = []
    binary_source = None
    data_provider_evidence = None
    interaction_evidence = None
    interaction_artifacts = []
    interaction_events = []
    interaction_log_bytes = 0
    terminal_input_count = 0
    def retain_interaction(filename, body):
        with open(directory / filename, 'xb') as artifact:
            artifact.write(body)
        interaction_artifacts.append({'file': filename, 'sha256': hashlib.sha256(body).hexdigest()})
    retain_interaction('python-identity.json', json.dumps(python_identity, sort_keys=True).encode())
    retain_interaction('parser-guards.json', json.dumps(parser_guard_evidence, sort_keys=True).encode())
    with open(directory / 'interaction-events.jsonl', 'xb'):
        pass
    def record_interaction(kind, **fields):
        global interaction_log_bytes
        event = {'ordinal': len(interaction_events), 'kind': kind,
                 'expected_source_commit': source, 'profile': profile, **fields}
        encoded = json.dumps(event, sort_keys=True).encode() + b'\n'
        interaction_log_bytes += len(encoded)
        assert interaction_log_bytes <= 1_000_000, 'bounded interaction log exceeded'
        with open(directory / 'interaction-events.jsonl', 'ab') as events:
            events.write(encoded)
        interaction_events.append(event)
    with open(directory / 'stderr.log', 'xb') as log:
        process = subprocess.Popen(
            [binary, '--storage', 'memory', '--bootstrap-ticks', '0', '--threads', '1',
             '--set', 'world_width=100', '--set', 'world_height=100',
             '--set', 'food_cell_size=50', '--set', 'population_minimum=0',
             '--set', 'persistence_interval=0', '--checkpoint-interval', '0'],
            stdin=slave, stdout=slave, stderr=log, cwd=directory, env=environment,
            preexec_fn=controlling_terminal)
        os.set_blocking(master, False)
        def read_terminal():
            try:
                chunk = os.read(master, 65536)
            except BlockingIOError:
                return
            assert chunk, 'terminal output reached EOF before its slave was closed'
            raw.extend(chunk)
            assert len(raw) <= 2_000_000, 'bounded PTY transcript exceeded'
            screen.feed(chunk)
        def send_terminal_input(data):
            global terminal_input_count
            assert 0 < len(data) <= 4096, 'bounded terminal input exceeded'
            deadline = time.monotonic() + 30
            written = 0
            while written < len(data) and time.monotonic() < deadline:
                assert process.poll() is None, 'terminal exited during input write'
                readable, writable, _ = select.select([master], [master], [], 0.05)
                if readable:
                    read_terminal()
                if writable:
                    try:
                        count = os.write(master, data[written:])
                    except BlockingIOError:
                        continue
                    assert count > 0, 'terminal input write made no progress'
                    written += count
            assert written == len(data), 'bounded terminal input write timed out'
            terminal_input_count += 1
            record_interaction('submitted_input', input_write=terminal_input_count,
                               bytes_hex=data.hex(), byte_count=written, transcript_offset=len(raw))
        def pump_until(predicate, description):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if predicate():
                    return
                if process.poll() is not None:
                    raise AssertionError('terminal exited during ' + description)
                if select.select([master], [], [], 0.05)[0]:
                    read_terminal()
            raise AssertionError('timeout during ' + description)
        try:
            pump_until(lambda: screen.alternate_entered, 'alternate-screen startup')
            pump_until(lambda: not (termios.tcgetattr(slave)[3] & termios.ICANON), 'raw-mode startup')
            raw_mode = not (termios.tcgetattr(slave)[3] & termios.ICANON)
            for key, title in routes:
                send_terminal_input(key.encode())
                pump_until(lambda: screen.alternate and not screen.synchronized
                           and title in screen.text(), title)
                observed.append(title)
                def screenshot_matches():
                    global binary_source
                    try:
                        request = urllib.request.Request(
                            'http://127.0.0.1:' + str(port) + '/api/screenshot/ascii',
                            headers={'User-Agent': 'OpenAI File Downloader, XaiImageApiFetch/1.0'})
                        with urllib.request.urlopen(
                                request, timeout=0.2) as response:
                            body = response.read(200_000)
                        terminal_text = screen.text()
                        if title.encode() in body and title in terminal_text and not screen.synchronized:
                            if key == '9':
                                child_source = re.search(r'\bBuild: ([0-9a-f]{40})\b', terminal_text)
                                rest_source = re.search(rb'\bBuild: ([0-9a-f]{40})\b', body)
                                if child_source is None or rest_source is None:
                                    return False
                                assert child_source[1] == source == rest_source[1].decode(), 'child build source mismatch'
                                binary_source = child_source[1]
                            terminal_bytes = terminal_text.encode()
                            with open(directory / ('screen-' + key + '.txt'), 'xb') as screenshot:
                                screenshot.write(body)
                            with open(directory / ('terminal-screen-' + key + '.txt'), 'xb') as terminal_capture:
                                terminal_capture.write(terminal_bytes)
                            screenshots.append({'route': title, 'sha256': hashlib.sha256(body).hexdigest(),
                                                'terminal_sha256': hashlib.sha256(terminal_bytes).hexdigest()})
                            return True
                    except urllib.error.HTTPError as error:
                        if error.code != 409:
                            raise
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    return False
                pump_until(screenshot_matches, 'presented screenshot for ' + title)
            if profile == 'truecolor':
                evidence_artifacts = []
                http_log_bytes = 0
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with open(directory / 'provider-http.jsonl', 'xb'):
                    pass

                def retain_response(filename, body):
                    with open(directory / filename, 'xb') as artifact:
                        artifact.write(body)
                    evidence_artifacts.append({'file': filename, 'sha256': hashlib.sha256(body).hexdigest()})

                def request_api(path, payload=None, allow_not_presented=False):
                    global http_log_bytes
                    request_body = None if payload is None else json.dumps(payload, sort_keys=True).encode()
                    headers = {'User-Agent': 'OpenAI File Downloader, XaiImageApiFetch/1.0'}
                    if request_body is not None:
                        headers['Content-Type'] = 'application/json'
                    request = urllib.request.Request('http://127.0.0.1:' + str(port) + path,
                                                     data=request_body, headers=headers)
                    try:
                        response = opener.open(request, timeout=2)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        body = response.read(200_001)
                        assert len(body) <= 200_000, 'bounded API response exceeded: ' + path
                        record = json.dumps({'method': request.get_method(), 'path': path,
                                             'status': response.status, 'headers': response.getheaders(),
                                             'request_body': None if request_body is None else request_body.decode(),
                                             'response_body': body.decode()}).encode() + b'\n'
                        http_log_bytes += len(record)
                        assert http_log_bytes <= 8_000_000, 'bounded provider HTTP evidence exceeded'
                        with open(directory / 'provider-http.jsonl', 'ab') as http_log:
                            http_log.write(record)
                        if allow_not_presented and response.status == 409:
                            return None
                        expected_status = 201 if path == '/api/v1/experiments' and payload is not None else 200
                        assert response.status == expected_status, (path, response.status, body.decode())
                    return body

                def applied_receipt(admission, filename):
                    command_id = admission['command_id']
                    def applied():
                        body = request_api('/api/control/status/' + command_id)
                        receipt = json.loads(body)
                        assert receipt['command_id'] == command_id, 'owner returned a different command identity'
                        assert receipt['application_state'] not in ('rejected', 'failed'), receipt
                        if receipt['application_state'] == 'applied':
                            retain_response(filename, body)
                            applied.result = receipt
                            return True
                        return False
                    pump_until(applied, 'actual owner application of ' + command_id)
                    return applied.result

                def header_state(text):
                    first_row = text.splitlines()[0] if text else ''
                    match = re.search(r'\| Tick: (\d+) \| Pop: \d+ \| (PAUSED|RUNNING)\b', first_row)
                    return None if match is None else {'tick': int(match[1]), 'paused': match[2] == 'PAUSED'}

                def owner_publication(command_id, filename, matches):
                    def published():
                        body = request_api('/api/status')
                        owner = json.loads(body)
                        if owner['last_applied_command'] != command_id or not matches(owner):
                            return False
                        retain_response(filename, body)
                        published.result = owner
                        return True
                    pump_until(published, 'actual owner publication for ' + command_id)
                    return published.result

                def capture_provider(prefix, title, matches):
                    def capture():
                        terminal_text = screen.text()
                        if not screen.alternate or screen.synchronized or title not in terminal_text or not matches(terminal_text):
                            return False
                        body = request_api('/api/screenshot/ascii', allow_not_presented=True)
                        if body is None:
                            return False
                        presented_text = body.decode()
                        if title not in presented_text or not matches(presented_text):
                            return False
                        retain_response(prefix + '-terminal.txt', terminal_text.encode())
                        retain_response(prefix + '-screen.txt', body)
                        return True
                    pump_until(capture, 'actual terminal and presented capture for ' + prefix)

                # REST owner changes cannot wake the UI by supplying keyboard
                # input. Its real Every subscription must observe the publication.
                input_count_before = terminal_input_count
                pause_body = request_api('/api/control/pause', {})
                retain_response('refresh-pause-response.json', pause_body)
                pause_receipt = applied_receipt(json.loads(pause_body), 'refresh-pause-receipt.json')
                before = owner_publication(pause_receipt['command_id'], 'refresh-before-owner.json',
                                           lambda owner: owner['paused'] is True)
                diagnostic_title = routes[-1][1]
                capture_provider('refresh-before', diagnostic_title,
                                 lambda text: header_state(text) == {'tick': before['tick'], 'paused': True})
                step_body = request_api('/api/control/step', {'count': 1})
                retain_response('refresh-step-response.json', step_body)
                step_receipt = applied_receipt(json.loads(step_body), 'refresh-step-receipt.json')
                after = owner_publication(step_receipt['command_id'], 'refresh-after-owner.json',
                                          lambda owner: owner['paused'] is True and owner['tick'] == before['tick'] + 1
                                          and owner['snapshot_revision'] > before['snapshot_revision'])
                assert step_receipt['scientific_revision'] > pause_receipt['scientific_revision'], (pause_receipt, step_receipt)
                capture_provider('refresh-after', diagnostic_title,
                                 lambda text: header_state(text) == {'tick': after['tick'], 'paused': True})
                input_count_during_change = terminal_input_count - input_count_before
                assert input_count_during_change == 0, 'terminal input masked the periodic refresh observation'

                # A checkpoint is created only by k through the live service
                # command. REST observes the resulting catalog; it never seeds it.
                send_terminal_input(b'7')
                replay_title = routes[3][1]
                pump_until(lambda: screen.alternate and not screen.synchronized
                           and replay_title in screen.text(), 'Replay before checkpoint creation')
                catalog_before_body = request_api('/api/v1/checkpoints?limit=32')
                retain_response('checkpoint-catalog-before.json', catalog_before_body)
                catalog_before = json.loads(catalog_before_body)
                assert catalog_before['total'] == len(catalog_before['items']) == 0, catalog_before
                before_ids = {item['checkpoint_id'] for item in catalog_before['items']}
                send_terminal_input(b'k')
                def checkpoint_created():
                    body = request_api('/api/v1/checkpoints?limit=32')
                    catalog = json.loads(body)
                    new_checkpoints = [item for item in catalog['items'] if item['checkpoint_id'] not in before_ids]
                    if not new_checkpoints:
                        return False
                    assert catalog['total'] == len(catalog['items']) == 1, catalog
                    assert len(new_checkpoints) == 1, catalog
                    retain_response('checkpoint-catalog-after.json', body)
                    checkpoint_created.result = new_checkpoints[0]
                    checkpoint_created.new_count = len(new_checkpoints)
                    return True
                pump_until(checkpoint_created, 'terminal-created checkpoint in actual REST catalog')
                checkpoint = checkpoint_created.result
                assert checkpoint['description'] == 'Interactive terminal checkpoint', checkpoint
                assert checkpoint['tick'] == after['tick'] and checkpoint['byte_size'] > 0, checkpoint
                checkpoint_body = request_api('/api/v1/checkpoints/' + checkpoint['checkpoint_id'])
                retain_response('checkpoint-metadata.json', checkpoint_body)
                assert json.loads(checkpoint_body) == checkpoint, 'catalog and checkpoint lookup disagree'
                checkpoint_row = re.compile(r'^\s*>\s+' + re.escape(checkpoint['checkpoint_id'])
                                            + r'\s+' + str(checkpoint['tick']) + r'\s', re.MULTILINE)
                checkpoint_status = 'Checkpoint ' + checkpoint['checkpoint_id'] + ' recorded at tick ' + str(checkpoint['tick'])
                checkpoint_receipt = '[APPLIED] journal=unobserved CreateCheckpoint: applied id=checkpoint-request-1'
                capture_provider('checkpoint-positive', replay_title,
                                 lambda text: checkpoint_row.search(text) is not None
                                 and checkpoint_receipt in text
                                 and checkpoint_status in text)

                # The existing runner seeds 16 registered MLP founders. Bound
                # its actual three-tick arm without inventing a zero-founder API.
                experiment_request = {'description': 'Live terminal data provider probe',
                                      'variants': [{'variant_id': 'pty_mlp', 'brain_family': 'mlp',
                                                    'config_overrides': {'world_width': 512,
                                                                         'world_height': 512,
                                                                         'food_cell_size': 128,
                                                                         'population_minimum': 0,
                                                                         'population_spawn_interval': 0}}],
                                      'seeds': [41], 'ticks_per_run': 3, 'max_concurrency': 1}
                retain_response('experiment-request.json', json.dumps(experiment_request, sort_keys=True).encode())
                experiment_created_body = request_api('/api/v1/experiments', experiment_request)
                retain_response('experiment-created.json', experiment_created_body)
                experiment_id = json.loads(experiment_created_body)['experiment_id']
                assert experiment_id, 'experiment creation omitted its identity'
                def experiment_completed():
                    body = request_api('/api/v1/experiments/' + experiment_id)
                    status = json.loads(body)
                    assert status['experiment_id'] == experiment_id, status
                    assert status['failed_runs'] == 0 and status['status'] not in ('failed', 'cancelled', 'stopped'), status
                    assert status['error_reason'] is None, status
                    if status['status'] != 'completed':
                        return False
                    assert status['total_runs'] == status['completed_runs'] == len(status['runs']) == 1, status
                    run = status['runs'][0]
                    assert run['variant_id'] == 'pty_mlp' and run['brain_family'] == 'mlp.baseline', run
                    assert run['state'] == 'completed' and run['seed'] == 41 and run['total_ticks'] == 3, run
                    assert re.fullmatch(r'[0-9a-f]{64}', run['final_digest']) and run['bundle_path'], run
                    assert run['error_reason'] is None, run
                    retain_response('experiment-completed.json', body)
                    experiment_completed.result = status
                    return True
                pump_until(experiment_completed, 'actual completed three-tick MLP experiment')
                experiment = experiment_completed.result
                run = experiment['runs'][0]
                send_terminal_input(b'6')
                experiment_title = routes[2][1]
                batch_row = re.compile(r'Experiment ID:\s*' + re.escape(experiment_id) + r'\s+Batch Status: completed\b')
                arm_row = re.compile(r'^\s*>\s+' + re.escape(run['variant_id']) + r'\s+'
                                     + re.escape(run['brain_family']) + r'\s+' + str(run['seed'])
                                     + r'\s+' + str(run['total_ticks']) + '/' + str(run['total_ticks']) + r'\s', re.MULTILINE)
                capture_provider('experiment-positive', experiment_title,
                                 lambda text: batch_row.search(text) is not None and arm_row.search(text) is not None)
                evidence_artifacts.append({'file': 'provider-http.jsonl',
                                           'sha256': hashlib.sha256((directory / 'provider-http.jsonl').read_bytes()).hexdigest()})
                data_provider_evidence = {
                    'checkpoint': {'checkpoint_id': checkpoint['checkpoint_id'], 'tick': checkpoint['tick'],
                                   'byte_size': checkpoint['byte_size'], 'checksum_sha256': checkpoint['checksum_sha256'],
                                   'description': checkpoint['description'], 'new_catalog_entries': checkpoint_created.new_count,
                                   'visible_application_receipt': checkpoint_receipt},
                    'experiment': {'experiment_id': experiment_id, 'status': experiment['status'],
                                   'total_runs': experiment['total_runs'], 'completed_runs': experiment['completed_runs'],
                                   'failed_runs': experiment['failed_runs'], 'run': run},
                    'refresh': {'before_tick': before['tick'], 'after_tick': after['tick'],
                                'before_snapshot_revision': before['snapshot_revision'],
                                'after_snapshot_revision': after['snapshot_revision'],
                                'paused_before': before['paused'], 'paused_after': after['paused'],
                                'pause_receipt': pause_receipt, 'step_receipt': step_receipt,
                                'terminal_input_count_during_change': input_count_during_change},
                    'artifacts': evidence_artifacts}

            # These bounded byte-input probes are not the companion's seeded
            # >=1000-event storm, capability matrix, or styled-cell comparison.
            interaction_http_bytes = 0
            interaction_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with open(directory / 'interaction-http.jsonl', 'xb'):
                pass

            def interaction_api(path, payload=None):
                global interaction_http_bytes
                request_body = None if payload is None else json.dumps(payload, sort_keys=True).encode()
                headers = {'User-Agent': 'OpenAI File Downloader, XaiImageApiFetch/1.0'}
                if request_body is not None:
                    headers['Content-Type'] = 'application/json'
                request = urllib.request.Request('http://127.0.0.1:' + str(port) + path,
                                                 data=request_body, headers=headers)
                try:
                    response = interaction_opener.open(request, timeout=2)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    body = response.read(200_001)
                    assert len(body) <= 200_000, 'bounded interaction response exceeded'
                    response_headers = {key.lower(): value for key, value in response.getheaders()}
                    record = json.dumps({'method': request.get_method(), 'path': path,
                                         'status': response.status, 'headers': response.getheaders(),
                                         'request_body': None if request_body is None else request_body.decode(),
                                         'response_body': body.decode()}).encode() + b'\n'
                    interaction_http_bytes += len(record)
                    assert interaction_http_bytes <= 8_000_000, 'bounded interaction HTTP log exceeded'
                    with open(directory / 'interaction-http.jsonl', 'ab') as http_log:
                        http_log.write(record)
                    if path == '/api/screenshot/ascii' and response.status == 409:
                        return None, response_headers
                    assert response.status == 200, (path, response.status, body.decode())
                return body, response_headers

            pause_body, _ = interaction_api('/api/control/pause', {})
            retain_interaction('interaction-pause-response.json', pause_body)
            pause_id = json.loads(pause_body)['command_id']
            def interaction_paused():
                receipt_body, _ = interaction_api('/api/control/status/' + pause_id)
                receipt = json.loads(receipt_body)
                assert receipt['command_id'] == pause_id, receipt
                assert receipt['application_state'] not in ('rejected', 'failed'), receipt
                if receipt['application_state'] != 'applied':
                    return False
                owner_body, _ = interaction_api('/api/status')
                owner = json.loads(owner_body)
                if owner['paused'] is not True or owner['last_applied_command'] != pause_id:
                    return False
                retain_interaction('interaction-pause-receipt.json', receipt_body)
                retain_interaction('interaction-before-owner.json', owner_body)
                interaction_paused.owner = owner
                interaction_paused.receipt = receipt
                return True
            pump_until(interaction_paused, 'owner paused before byte input and resize')
            interaction_before = interaction_paused.owner
            baseline_tick = interaction_before['tick']

            def capture_interaction(prefix, title, matches=lambda text: True, output_after=None,
                                    require_footer_geometry=False):
                def capture():
                    terminal_text = screen.text()
                    if (not screen.alternate or screen.synchronized or screen.pending
                            or title not in terminal_text or not matches(terminal_text)
                            or (output_after is not None and len(raw) <= output_after)):
                        return False
                    body, headers = interaction_api('/api/screenshot/ascii')
                    if body is None:
                        return False
                    presented_text = body.decode()
                    if (headers.get('x-scriptbots-frame-size') != str(screen.width) + 'x' + str(screen.height)
                            or headers.get('x-scriptbots-frame-tick') != str(baseline_tick)
                            or title not in presented_text or not matches(presented_text)):
                        return False
                    assert len(presented_text.splitlines()) == screen.height, 'presented row count disagrees with viewport'
                    assert len(screen.cells) == screen.height and all(len(row) == screen.width for row in screen.cells)
                    if require_footer_geometry:
                        terminal_geometry = diagnostics_footer_geometry(terminal_text, screen.width, screen.height)
                        presented_geometry = diagnostics_footer_geometry(presented_text, screen.width, screen.height)
                        if terminal_geometry is None or terminal_geometry != presented_geometry:
                            return False
                    terminal_bytes = terminal_text.encode()
                    retain_interaction(prefix + '-terminal.txt', terminal_bytes)
                    retain_interaction(prefix + '-screen.txt', body)
                    retain_interaction(prefix + '-headers.json', json.dumps(headers, sort_keys=True).encode())
                    capture.result = {'probe': prefix, 'width': screen.width, 'height': screen.height,
                                      'presented_tick': int(headers['x-scriptbots-frame-tick']),
                                      'presented_frame_size': headers['x-scriptbots-frame-size'],
                                      'terminal_sha256': hashlib.sha256(terminal_bytes).hexdigest(),
                                      'presented_sha256': hashlib.sha256(body).hexdigest(),
                                      'transcript_offset': len(raw)}
                    if require_footer_geometry:
                        capture.result['footer_geometry'] = {'terminal': terminal_geometry,
                                                             'presented': presented_geometry}
                    record_interaction('observed_capture', **capture.result)
                    return True
                pump_until(capture, 'serialized terminal and presented REST observation for ' + prefix)
                return capture.result

            def paused_header(text):
                first_row = text.splitlines()[0] if text else ''
                return re.search(r'\| Tick: ' + str(baseline_tick) + r' \| Pop: \d+ \| PAUSED\b', first_row) is not None

            diagnostic_title = routes[-1][1]
            send_terminal_input(b'9')
            capture_interaction('interaction-before', diagnostic_title, paused_header,
                                require_footer_geometry=True)
            foreground_group = struct.unpack('i', fcntl.ioctl(master, termios.TIOCGPGRP, struct.pack('i', 0)))[0]
            assert foreground_group == process.pid, 'child does not own its controlling terminal foreground group'
            resize_ladder = [(40, 12), (60, 20), (80, 36), (120, 40), (160, 50), (120, 36)]
            observed_resizes = []
            for index, (width, height) in enumerate(resize_ladder):
                # Drain available old-size output without discarding a partial
                # escape sequence or resetting the incremental UTF-8 decoder.
                drain_deadline = time.monotonic() + 0.2
                while time.monotonic() < drain_deadline and select.select([master], [], [], 0)[0]:
                    read_terminal()
                pump_until(lambda: not screen.synchronized and not screen.pending
                                   and diagnostics_footer_geometry(screen.text(), screen.width, screen.height) is not None,
                           'complete old-size footer before resize control')
                prefix = 'resize-' + str(index)
                before_bytes = screen.text().encode()
                before_file = prefix + '-before-terminal.txt'
                retain_interaction(before_file, before_bytes)
                # Deliberately omit child repainting in this synthetic fault
                # control. Its cells come from the actual old-size terminal;
                # local clamp/pad plus a valid harmless SGR reset must fail the
                # identical footer oracle used for the live new-size capture.
                stale = TerminalScreen(screen.width, screen.height)
                stale.cells = [row[:] for row in screen.cells]
                stale.x, stale.y, stale.saved = screen.x, screen.y, screen.saved
                stale.scroll_top, stale.scroll_bottom = screen.scroll_top, screen.scroll_bottom
                stale.wrap_pending, stale.autowrap = screen.wrap_pending, screen.autowrap
                before_dimensions = (screen.width, screen.height)
                stale.resize(width, height)
                benign_input = b'\x1b[0m'
                stale.feed(benign_input, final=True)
                stale_bytes = stale.text().encode()
                stale_file = prefix + '-stale-terminal.txt'
                retain_interaction(stale_file, stale_bytes)
                dimensions_observed = (stale.width, stale.height) == (width, height)
                dimensions_observed &= len(stale.cells) == height and all(len(row) == width for row in stale.cells)
                heading_preserved = diagnostic_title in stale.text()
                stale_geometry = diagnostics_footer_geometry(stale.text(), width, height)
                assert dimensions_observed and heading_preserved, 'stale control must satisfy the old dimension/title checks'
                assert stale_geometry is None, 'local stale-cell resize plus harmless output satisfied footer oracle'
                offset = len(raw)
                started = time.monotonic()
                screen.resize(width, height)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', height, width, 0, 0))
                os.killpg(process.pid, signal.SIGWINCH)
                assert struct.unpack('HHHH', fcntl.ioctl(slave, termios.TIOCGWINSZ, b'\0' * 8))[:2] == (height, width)
                record_interaction('requested_resize', width=width, height=height,
                                   signal='SIGWINCH', transcript_offset=offset)
                result = capture_interaction(prefix, diagnostic_title,
                                             paused_header if width >= 80 else lambda text: True,
                                             output_after=offset, require_footer_geometry=True)
                result['elapsed_ms'] = (time.monotonic() - started) * 1000
                result['serialized_bytes_observed'] = len(raw) - offset
                result['stale_cell_control'] = {
                    'synthetic_control': True,
                    'actual_before': {'width': before_dimensions[0], 'height': before_dimensions[1],
                                      'terminal_file': before_file,
                                      'terminal_sha256': hashlib.sha256(before_bytes).hexdigest()},
                    'benign_input_hex': benign_input.hex(), 'benign_input_byte_length': len(benign_input),
                    'parser_sequence_accepted': not stale.pending,
                    'terminal_file': stale_file, 'terminal_sha256': hashlib.sha256(stale_bytes).hexdigest(),
                    'requested_dimensions_observed': dimensions_observed, 'full_heading_preserved': heading_preserved,
                    'observed_footer_geometry': stale_geometry,
                    'expected_footer_geometry': result['footer_geometry']['presented'],
                    'rejected': stale_geometry is None}
                record_interaction('rejected_stale_resize_control', probe=prefix, **result['stale_cell_control'])
                observed_resizes.append(result)
                record_interaction('observed_resize', **result)

            def selected_palette_row(text):
                selected = []
                for row, line in enumerate(text.splitlines()):
                    match = re.search(r'> ((?:\[Recent\] )?\[[^\]]+\] [^│|\r\n]+)', line)
                    if match is not None:
                        selected.append((row, match[1].strip()))
                return selected[0] if len(selected) == 1 else None

            palette_title = 'Command Palette (type to filter, Esc to exit)'
            send_terminal_input(b'\x10')
            capture_interaction('palette-open', palette_title,
                                lambda text: selected_palette_row(text) is not None)
            initial_selection = selected_palette_row(screen.text())
            observed_arrows = []
            send_terminal_input(b'\x1b[B')
            def arrow_down_matches(text):
                selected = selected_palette_row(text)
                return selected is not None and selected[0] == initial_selection[0] + 1 and selected[1] != initial_selection[1]
            down_capture = capture_interaction('palette-arrow-down', palette_title, arrow_down_matches)
            down_selection = selected_palette_row(screen.text())
            observed_arrows.append({'key': 'down', 'input_hex': b'\x1b[B'.hex(),
                                    'before': initial_selection, 'after': down_selection, 'capture': down_capture})
            record_interaction('observed_arrow', **observed_arrows[-1])
            send_terminal_input(b'\x1b[A')
            up_capture = capture_interaction('palette-arrow-up', palette_title,
                                             lambda text: selected_palette_row(text) == initial_selection)
            observed_arrows.append({'key': 'up', 'input_hex': b'\x1b[A'.hex(),
                                    'before': down_selection, 'after': selected_palette_row(screen.text()),
                                    'capture': up_capture})
            record_interaction('observed_arrow', **observed_arrows[-1])

            send_terminal_input(b'Open Lineages')
            filtered_title = 'Command Palette: "Open Lineages"'
            target_label = '[Navigation] Open Lineages and Phylogeny'
            capture_interaction('palette-mouse-target', filtered_title,
                                lambda text: target_label in text and selected_palette_row(text) is not None)
            positions = [(column, row) for row, cells in enumerate(screen.cells)
                         for column in range(screen.width - len(target_label) + 1)
                         if ''.join(cells[column:column + len(target_label)]) == target_label]
            assert len(positions) == 1, 'mouse target must occur once in the actual visible cells'
            column, row = positions[0]
            mouse_column, mouse_row = column + 4 + 1, row + 1
            capture_modes_before_input = sorted(screen.enabled_capture_modes)
            observed_sgr_capture = {1000, 1002, 1006}.issubset(screen.enabled_capture_modes)
            assert observed_sgr_capture, 'actual SGR button capture was not enabled before mouse input'
            down_bytes = ('\x1b[<0;' + str(mouse_column) + ';' + str(mouse_row) + 'M').encode()
            up_bytes = ('\x1b[<0;' + str(mouse_column) + ';' + str(mouse_row) + 'm').encode()
            send_terminal_input(down_bytes)
            capture_interaction('palette-mouse-down', filtered_title, lambda text: target_label in text)
            send_terminal_input(up_bytes)
            navigation_capture = capture_interaction('palette-mouse-navigation', routes[0][1],
                                                     lambda text: paused_header(text) and 'Command Palette' not in text)
            mouse_navigation = {'protocol': 'SGR', 'column_one_based': mouse_column, 'row_one_based': mouse_row,
                                'observed_sgr_capture': observed_sgr_capture,
                                'capture_modes_before_input': capture_modes_before_input,
                                'visible_item': target_label, 'down_hex': down_bytes.hex(), 'up_hex': up_bytes.hex(),
                                'observed_route': 'Lineages', 'capture': navigation_capture}
            record_interaction('observed_mouse_navigation', **mouse_navigation)
            send_terminal_input(b'9')
            source_pattern = re.compile(r'\bBuild: ' + re.escape(source) + r'\b')
            responsiveness_capture = capture_interaction('interaction-responsive', diagnostic_title,
                                                          lambda text: paused_header(text) and source_pattern.search(text) is not None)
            owner_after_body, _ = interaction_api('/api/status')
            interaction_after = json.loads(owner_after_body)
            retain_interaction('interaction-after-owner.json', owner_after_body)
            assert interaction_after['tick'] == baseline_tick and interaction_after['paused'] is True, interaction_after
            assert interaction_after['last_applied_command'] == pause_id, 'UI input unexpectedly applied a host command'
            assert interaction_after['config_revision'] == interaction_before['config_revision'], 'UI navigation changed owner config'
            interaction_evidence = {
                'source_commit': binary_source,
                'requested_resize_sequence': resize_ladder,
                'observed_resize_sequence': [(result['width'], result['height']) for result in observed_resizes],
                'observed_resize_probes': observed_resizes, 'observed_arrow_probes': observed_arrows,
                'mouse_navigation': mouse_navigation, 'before_owner': interaction_before,
                'after_owner': interaction_after, 'pause_receipt': interaction_paused.receipt,
                'owner_tick_unchanged': interaction_after['tick'] == interaction_before['tick'],
                'responsiveness_observed': responsiveness_capture['presented_tick'] == baseline_tick
                                           and responsiveness_capture['presented_frame_size'] == '120x36',
                'responsiveness_capture': responsiveness_capture,
                'submitted_input_write_count_before_quit': len([event for event in interaction_events
                                                               if event['kind'] == 'submitted_input']),
                'observed_probe_count': len([event for event in interaction_events
                                             if event['kind'] in ('observed_resize', 'observed_arrow',
                                                                  'observed_mouse_navigation')])}
            retain_interaction('interaction-results.json', json.dumps(interaction_evidence, sort_keys=True).encode())
            send_terminal_input(b'q')
            deadline = time.monotonic() + 30
            while process.poll() is None and time.monotonic() < deadline:
                if select.select([master], [], [], 0.05)[0]:
                    read_terminal()
            if process.poll() is None:
                raise AssertionError('terminal quit did not finish ordered shutdown')
            while select.select([master], [], [], 0)[0]:
                read_terminal()
            screen.feed(b'', final=True)
            restored = termios.tcgetattr(slave) == initial
            alternate_restored = screen.alternate_left and not screen.alternate
            assert process.returncode == 0, process.returncode
            assert restored, 'termios attributes were not restored'
            assert alternate_restored, 'alternate screen was not restored'
            assert screen.cursor_hidden_observed, 'actual cursor hiding was not observed'
            assert screen.cursor_show_observed and screen.cursor_visible, 'cursor was not visible after quit'
            assert not screen.enabled_capture_modes, 'capture modes remained enabled after quit'
            assert binary_source == source, 'actual child build source was not observed'
            if profile == 'truecolor':
                assert screen.truecolor_count > 0, 'actual truecolor SGR was not emitted'
            else:
                assert screen.color_count == 0, 'NO_COLOR emitted non-default SGR colors'
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            with open(directory / 'terminal.raw', 'xb') as transcript:
                transcript.write(raw)
            with open(directory / 'terminal-attributes.json', 'x') as attributes:
                json.dump({'initial': repr(initial), 'final': repr(termios.tcgetattr(slave))}, attributes)
            os.close(master)
            os.close(slave)
    lifecycle_bytes = (directory / 'runtime.log').read_bytes()
    lifecycle = lifecycle_bytes.decode(errors='replace')
    finalized = 'terminal lifecycle finalized' in lifecycle and 'frankentui' in lifecycle
    assert finalized, 'real FrankenTUI shutdown callback was not observed'
    for filename in ('interaction-events.jsonl', 'interaction-http.jsonl'):
        interaction_artifacts.append({'file': filename,
                                      'sha256': hashlib.sha256((directory / filename).read_bytes()).hexdigest()})
    interaction_evidence['artifacts'] = interaction_artifacts
    print(json.dumps({'schema': 'scriptbots.live-tui-pty.v1', 'source_commit': binary_source,
                      'binary_source_observed': binary_source is not None,
                      'capability_profile': profile, 'observed_science_routes': observed,
                      'observed_screenshot_routes': screenshots,
                      'data_provider_evidence': data_provider_evidence,
                      'python_identity': python_identity,
                      'parser_guard_evidence': parser_guard_evidence,
                      'byte_input_resize_evidence': interaction_evidence,
                      'observed_color_sgr_count': screen.color_count,
                      'observed_truecolor_sgr_count': screen.truecolor_count,
                      'observed_style_sgr_count': screen.attribute_count,
                      'exit_code': process.returncode, 'raw_mode_observed': raw_mode,
                      'terminal_attributes_restored': restored,
                      'alternate_screen_restored': alternate_restored,
                      'cursor_hidden_observed': screen.cursor_hidden_observed,
                      'cursor_visible_at_exit': screen.cursor_show_observed and screen.cursor_visible,
                      'enabled_capture_modes_at_exit': sorted(screen.enabled_capture_modes),
                      'frankentui_shutdown_observed': finalized,
                      'runtime_log_sha256': hashlib.sha256(lifecycle_bytes).hexdigest(),
                      'transcript_sha256': hashlib.sha256(raw).hexdigest(),
                      'artifact_directory': str(directory)}), flush=True)
"#;

const HOST_SESSION: HostSessionId = HostSessionId::new(0x7e2e);

fn terminal_host(
    world: WorldState,
    persistence: Option<PersistenceAdmissionSession>,
    journal: Box<dyn JournalPort + Send>,
) -> Result<HostThread> {
    let persistence = match persistence {
        Some(session) => session,
        None => world.bind_persistence(Box::new(scriptbots_core::NullPersistence))?,
    };
    HostThread::spawn(
        HOST_SESSION,
        world,
        persistence,
        journal,
        HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: true,
                speed_multiplier: 1.0,
            },
            capture_agent_visuals: true,
            ..HostCoreOptions::default()
        },
        ChannelHostOptions::default(),
    )
}

struct EnvCleanup {
    keys: Vec<String>,
}

impl EnvCleanup {
    fn new() -> Self {
        Self { keys: Vec::new() }
    }

    fn set(&mut self, key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
        self.keys.push(key.to_string());
    }
}

impl Drop for EnvCleanup {
    fn drop(&mut self) {
        for key in &self.keys {
            unsafe {
                std::env::remove_var(key);
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct FrameStatsDto {
    tick: u64,
    epoch: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    avg_energy: f32,
    buffer: Option<HeadlessBufferEvidenceDto>,
}

#[derive(Debug, Deserialize)]
struct HeadlessBufferEvidenceDto {
    backend: String,
    capability_profile: String,
    viewport_width: u16,
    viewport_height: u16,
    current_tick: u64,
    non_blank_cells: usize,
    styled_cells: usize,
    skipped_cells: usize,
    forced_width_cells: usize,
    empty_symbol_cells: usize,
    full_cell_fnv1a64: String,
    semantic_regions: Vec<String>,
    /// Per-panel evidence added by bd-2z0.14.2.6. Additive to `semantic_regions`,
    /// which still names the regions; this proves each one separately so a broken
    /// chart is distinguishable from a broken map.
    regions: Vec<RegionEvidenceDto>,
}

#[derive(Debug, Deserialize)]
struct RegionEvidenceDto {
    name: String,
    #[allow(dead_code)]
    x: u16,
    #[allow(dead_code)]
    y: u16,
    width: u16,
    height: u16,
    has_room: bool,
    marker_present: bool,
    non_blank_cells: usize,
    fnv1a64: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ReportSummaryDto {
    frame_count: usize,
    ticks_simulated: u64,
    final_tick: u64,
    final_epoch: u64,
    final_agent_count: usize,
    total_births: usize,
    total_deaths: usize,
    avg_energy_mean: f32,
    avg_energy_min: f32,
    avg_energy_max: f32,
}

#[derive(Debug, Deserialize)]
struct HeadlessReportDto {
    initial: FrameStatsDto,
    frames: Vec<FrameStatsDto>,
    summary: ReportSummaryDto,
}

fn assert_test_backend_buffer_evidence(report: &HeadlessReportDto) {
    assert!(
        report.initial.buffer.is_none(),
        "the initial measurement precedes the first rendered frame"
    );
    for frame in &report.frames {
        let buffer = frame
            .buffer
            .as_ref()
            .expect("every TestBackend frame must retain rendered-buffer evidence");
        assert_eq!(buffer.backend, "ratatui_test_backend");
        assert_eq!(
            buffer.capability_profile, "ascii_natural_fixed_80x36",
            "headless evidence must identify its deterministic non-live capability profile"
        );
        assert_eq!((buffer.viewport_width, buffer.viewport_height), (80, 36));
        assert_eq!(
            buffer.current_tick, frame.tick,
            "rendered tick text must describe the same snapshot as the frame report"
        );
        assert!(
            buffer.non_blank_cells > 100,
            "the TestBackend frame must contain substantive rendered content"
        );
        assert!(
            buffer.styled_cells > 0,
            "the full-cell proof must cover styles"
        );
        assert_eq!(
            buffer.skipped_cells, 0,
            "the ASCII TestBackend profile must not skip cells"
        );
        assert_eq!(
            buffer.forced_width_cells, 0,
            "the ASCII TestBackend profile must not require forced-width cells"
        );
        assert_eq!(
            buffer.empty_symbol_cells, 0,
            "the ASCII TestBackend profile must fill each cell symbol"
        );
        assert_eq!(
            buffer.full_cell_fnv1a64.len(),
            16,
            "buffer evidence must include the full deterministic FNV-1a digest"
        );
        // SCHEMA CHANGE, bd-2z0.14.2.6. This was the four whole-frame needles the
        // inspector searched for (terminal_hud, current_tick, world_map,
        // vital_stats). It is now the panel set derived from the frame's own
        // layout. `current_tick` left the list because it was never a panel — it is
        // a frame-wide string and the inspector still asserts it as one.
        assert_eq!(
            buffer.semantic_regions,
            [
                "header",
                "rail",
                "world_map",
                "vital_stats",
                "trends",
                "leaderboard",
                "oldest",
                "insights",
                "brains",
                "events",
            ],
            "buffer evidence must come from the expected terminal HUD regions"
        );
        assert_eq!(
            buffer.regions.len(),
            buffer.semantic_regions.len(),
            "every named region must carry its own evidence: {:?}",
            buffer.regions
        );

        let mut region_hashes = std::collections::HashSet::new();
        for region in &buffer.regions {
            assert_eq!(
                region.fnv1a64.len(),
                16,
                "region {} must carry a full FNV-1a digest",
                region.name
            );
            assert!(
                region.width > 0 && region.height > 0,
                "region {} must report the rectangle it was hashed over",
                region.name
            );
            // At the 80x36 evidence viewport every panel fits, so every region must
            // both have room and have shown its marker. A report where a panel
            // quietly went absent at this size is a layout regression, and without
            // this assertion it would be recorded rather than caught.
            assert!(
                region.has_room,
                "region {} must have room at 80x36: {region:?}",
                region.name
            );
            assert!(
                region.marker_present,
                "region {} had room and must have drawn its marker: {region:?}",
                region.name
            );
            assert!(
                region.non_blank_cells > 0,
                "region {} must have painted something: {region:?}",
                region.name
            );
            assert!(
                region_hashes.insert(region.fnv1a64.clone()),
                "region hashes must be distinct, or they are not per-region: {} \
                 duplicates an earlier digest",
                region.name
            );
        }
    }
}

type HttpResponse = (u16, Vec<(String, String)>, String);

/// Minimal HTTP/1.1 GET over a real socket.
///
/// Deliberately hand-rolled rather than adding an HTTP client dependency: the
/// request under test is a bare GET, and a new third-party dev-dependency is an
/// operator decision this proof does not need. Returns `(status, headers, body)`.
fn http_get(addr: std::net::SocketAddr, path: &str) -> Result<HttpResponse> {
    use std::io::{Read, Write};

    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(20)))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).into_owned();

    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP response: {text:?}"))?;
    let mut lines = head.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty HTTP response"))?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("no status code in {status_line:?}"))?
        .parse()?;
    let headers = lines
        .filter_map(|line| {
            line.split_once(':')
                .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        })
        .collect();
    Ok((status, headers, body.to_string()))
}

/// MOCK-FREE HEADLESS + REST EVIDENCE, end to end.
///
/// Nothing here is stubbed: a real world, the real terminal renderer driving real
/// frames, the real control runtime with a real bound REST listener, and a real
/// HTTP request over a real socket. The single structured evidence line it prints
/// is what `scripts/e2e_tui_evidence.sh` gates on.
///
/// WHAT THIS PROVES THAT THE UNIT SUITE CANNOT: that the frame the renderer
/// PUBLISHED is the frame the SERVER SERVED, across a process-internal boundary
/// that the unit tests reach around. The unit test for the handler feeds it a
/// buffer directly; this one never touches the slot — the renderer fills it as a
/// side effect of drawing, and the bytes come back off a socket.
///
/// THE NEGATIVE CONTROL IS DELIBERATELY NOT HERE, and the reason is recorded rather
/// than glossed: "a deliberately broken widget must fail only its expected region"
/// needs a widget that draws wrongly, and nothing outside the binary can cause that
/// — every external lever (theme, palette, capability, reduced motion, viewport)
/// changes the frame LEGITIMATELY. Faking it would require a test-only breakage
/// hook in production paint code. The localization property is proven at the buffer
/// level instead, by `a_broken_widget_fails_its_own_region_and_no_others`, which
/// blanks one panel's rectangle and requires exactly one region hash to move. The
/// script runs that test as part of this gate so the two halves are executed
/// together even though they live at different levels.
#[test]
#[serial]
fn tui_evidence_e2e_serves_the_frame_the_renderer_published() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let frames = 8usize;
    let seed: u64 = 0x5C81_B075;
    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("tui_evidence_report.json");

    let mut env = EnvCleanup::new();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS", "1");
    let frames_env = frames.to_string();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", &frames_env);
    let report_env = report_path.to_string_lossy().into_owned();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_env);

    let world = WorldState::new(ScriptBotsConfig {
        // 160/32 and 96/32 both divide: the config refuses a world whose extent is
        // not a whole number of food cells.
        food_cell_size: 32,
        world_width: 160,
        world_height: 96,
        population_minimum: 0,
        population_spawn_interval: 0,
        rng_seed: Some(seed),
        persistence_interval: 0,
        ..ScriptBotsConfig::default()
    })?;
    let host = terminal_host(world, None, Box::new(VolatileJournal::default()))?;

    // Port 0: the OS picks, and the reservation reports what it bound. A fixed port
    // would collide with whatever else is running on a shared machine.
    let control_config = ControlServerConfig {
        rest_enabled: true,
        rest_address: "127.0.0.1:0".parse()?,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };
    let reservation = scriptbots_app::ControlServerReservation::prepare(control_config)?;
    let rest_addr = reservation
        .rest_address()
        .ok_or_else(|| anyhow::anyhow!("REST listener was not bound"))?;
    let (control_runtime, command_submit) = reservation.launch(host.port())?;

    let renderer = TerminalRenderer::default();
    {
        let context = RendererContext {
            host: host.port(),
            analytics: AnalyticsSnapshotProvider::empty(),
            control_runtime: &control_runtime,
            command_submit,
            scenario: Arc::new(ScenarioIdentityV0::caller_seeded("tui-evidence-e2e")),
        };
        renderer.run(context)?;
    }

    // The runtime is still up: the headless run has finished and published its last
    // frame, so there is no race between drawing and fetching.
    let (status, headers, body) = http_get(rest_addr, "/api/screenshot/ascii")?;
    assert_eq!(
        status, 200,
        "the endpoint must serve the presented frame, got {status} with body {body:?}"
    );

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    let last = report
        .frames
        .last()
        .and_then(|frame| frame.buffer.as_ref())
        .ok_or_else(|| anyhow::anyhow!("headless report retained no buffer evidence"))?;

    let served_size = headers
        .iter()
        .find(|(key, _)| key == "x-scriptbots-frame-size")
        .map(|(_, value)| value.clone())
        .unwrap_or_default();
    assert_eq!(
        served_size,
        format!("{}x{}", last.viewport_width, last.viewport_height),
        "the served frame must have the same geometry the report recorded"
    );
    // Non-vacuity: an empty body would match an empty expectation.
    let served_lines = body.lines().count();
    assert_eq!(
        served_lines,
        usize::from(last.viewport_height),
        "the served text must carry one line per buffer row"
    );
    assert!(
        body.contains("ScriptBots"),
        "the served frame must contain the HUD header it was drawn with"
    );

    control_runtime.shutdown()?;
    let final_owner = host.join()?;
    assert_eq!(final_owner.snapshot.world.tick, frames as u64);

    // The structured evidence line the script gates on. Every field the acceptance
    // criterion names, so a reviewer can audit the run without re-running it.
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let regions: Vec<String> = last
        .regions
        .iter()
        .map(|region| {
            format!(
                "{{\"name\":\"{}\",\"rect\":\"{}x{}+{}+{}\",\"room\":{},\"marker\":{},\"nonblank\":{},\"hash\":\"{}\"}}",
                region.name,
                region.width,
                region.height,
                region.x,
                region.y,
                region.has_room,
                region.marker_present,
                region.non_blank_cells,
                region.fnv1a64
            )
        })
        .collect();
    println!(
        "{{\"schema\":\"scriptbots.tui-evidence.v1\",\"seed\":{seed},\"frames\":{frames},\
         \"viewport\":\"{}x{}\",\"capability_profile\":\"{}\",\"backend\":\"{}\",\
         \"endpoint\":\"/api/screenshot/ascii\",\"endpoint_status\":{status},\
         \"report_path\":\"{}\",\"served_bytes\":{},\"served_lines\":{served_lines},\
         \"full_cell_fnv1a64\":\"{}\",\"region_count\":{},\"regions\":[{}],\
         \"source_commit\":\"{commit}\"}}",
        last.viewport_width,
        last.viewport_height,
        last.capability_profile,
        last.backend,
        report_path.display(),
        body.len(),
        last.full_cell_fnv1a64,
        regions.len(),
        regions.join(","),
    );

    Ok(())
}

#[test]
#[serial]
fn terminal_test_backend_generates_semantic_buffer_report() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn,scriptbots_app=info")
        .with_test_writer()
        .try_init();

    let frames = 32usize;

    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("terminal_report.json");

    let mut env = EnvCleanup::new();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS", "1");
    let frames_env = frames.to_string();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", &frames_env);
    let report_env = report_path.to_string_lossy().into_owned();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_env);

    let mut config = ScriptBotsConfig {
        food_cell_size: 32,
        world_width: 160,
        world_height: 96,
        population_minimum: 0,
        population_spawn_interval: 0,
        history_capacity: 512,
        rng_seed: Some(0xDEC0_DEAD),
        initial_food: 0.35,
        food_respawn_interval: 6,
        food_respawn_amount: 0.45,
        food_max: 1.0,
        food_growth_rate: 0.18,
        food_decay_rate: 0.0008,
        food_diffusion_rate: 0.18,
        reproduction_cooldown: 12,
        reproduction_rate_herbivore: 2.5,
        reproduction_rate_carnivore: 2.5,
        reproduction_energy_cost: 0.12,
        reproduction_child_energy: 0.9,
        reproduction_spawn_jitter: 12.0,
        reproduction_spawn_back_distance: 6.0,
        reproduction_partner_chance: 0.4,
        metabolism_drain: 0.006,
        movement_drain: 0.012,
        temperature_discomfort_rate: 0.0015,
        food_intake_rate: 0.008,
        food_waste_rate: 0.0005,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 1.0,
        ..ScriptBotsConfig::default()
    };
    config.food_sharing_rate = 0.15;
    config.food_transfer_rate = 0.0025;

    let mut world = WorldState::new(config.clone())?;
    let mut rng = SmallRng::seed_from_u64(0xBAD5_EED5);
    for index in 0..32 {
        let position = Position::new(
            rng.random_range(0.0..config.world_width as f32),
            rng.random_range(0.0..config.world_height as f32),
        );
        let heading = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let color = [
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
        ];
        let agent = AgentData::new(
            position,
            Velocity::default(),
            heading,
            if index < 4 { -0.01 } else { 1.0 },
            color,
            0.1,
            false,
            0,
            Generation::default(),
        );
        world
            .try_spawn_agent(agent)
            .expect("terminal fixture agent is finite");
    }
    let host = terminal_host(world, None, Box::new(VolatileJournal::default()))?;

    let analytics = AnalyticsSnapshotProvider::empty();

    let control_config = ControlServerConfig {
        rest_enabled: false,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };

    let (control_runtime, command_submit) = ControlRuntime::launch(host.port(), control_config)?;

    let renderer = TerminalRenderer::default();
    {
        let context = RendererContext {
            host: host.port(),
            analytics: analytics.clone(),
            control_runtime: &control_runtime,
            command_submit,
            scenario: Arc::new(ScenarioIdentityV0::caller_seeded("e2e-scenario")),
        };
        renderer.run(context)?;
    }

    control_runtime.shutdown()?;
    let final_owner = host.join()?;

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    assert_test_backend_buffer_evidence(&report);
    let summary = &report.summary;

    assert_eq!(
        summary.frame_count, frames,
        "headless renderer should honour requested frame budget"
    );
    assert_eq!(
        summary.ticks_simulated,
        summary.final_tick.saturating_sub(report.initial.tick),
        "tick delta should align with simulated frames"
    );
    assert!(
        summary.final_agent_count > 0,
        "simulation should retain surviving agents"
    );
    assert!(
        summary.avg_energy_max > summary.avg_energy_min,
        "energy extrema should be well ordered and non-degenerate"
    );

    let total_births: usize = report.frames.iter().map(|frame| frame.births).sum();
    let total_deaths: usize = report.frames.iter().map(|frame| frame.deaths).sum();
    assert_eq!(
        total_births, summary.total_births,
        "summary births should match frame-wise birth totals"
    );
    assert_eq!(
        total_deaths, summary.total_deaths,
        "summary deaths should match frame-wise death totals"
    );
    assert!(
        summary.total_births > 8,
        "simulation should exhibit meaningful reproduction activity (births={})",
        summary.total_births
    );
    assert!(
        summary.total_deaths > 0,
        "simulation should register at least one death"
    );

    let frames_with_births = report
        .frames
        .iter()
        .filter(|frame| frame.births > 0)
        .count();
    assert!(
        frames_with_births >= 6,
        "births should occur across multiple frames (frames_with_births={frames_with_births})"
    );

    let frames_with_deaths = report
        .frames
        .iter()
        .filter(|frame| frame.deaths > 0)
        .count();
    assert!(
        frames_with_deaths >= 1,
        "deaths should appear in the run (frames_with_deaths={frames_with_deaths})"
    );

    let agent_counts: Vec<usize> = report
        .frames
        .iter()
        .map(|frame| frame.agent_count)
        .collect();
    let min_agents = *agent_counts.iter().min().expect("min agent count");
    let max_agents = *agent_counts.iter().max().expect("max agent count");
    assert!(
        max_agents > min_agents,
        "agent count should vary over the run (min={min_agents}, max={max_agents})"
    );

    assert_eq!(
        report.initial.agent_count + summary.total_births - summary.total_deaths,
        summary.final_agent_count,
        "agent conservation should hold (initial + births - deaths = final)"
    );

    {
        let snapshot = &final_owner.snapshot;
        assert_eq!(
            snapshot.world.tick, summary.final_tick,
            "world tick should advance to the reported final tick"
        );
        let history = &snapshot.summary_history;
        assert!(
            history.len() >= frames,
            "world history should retain per-tick summaries (len={})",
            history.len()
        );
        assert!(
            history.iter().any(|entry| entry.births > 0),
            "world history should record at least one birth"
        );
        assert!(
            history.iter().any(|entry| entry.deaths > 0),
            "world history should record at least one death"
        );
    }

    Ok(())
}

#[test]
#[serial]
fn terminal_test_backend_applies_control_updates_and_renders_receipts() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn,scriptbots_app=info")
        .with_test_writer()
        .try_init();

    let frames = 37usize;

    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("terminal_control_report.json");

    let storage_dir = tempdir()?;
    let storage_path = storage_dir.path().join("terminal_control.sqlite");

    let mut env = EnvCleanup::new();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS", "1");
    let frames_env = frames.to_string();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", &frames_env);
    let report_env = report_path.to_string_lossy().into_owned();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_env);
    env.set("RUST_LOG", "warn,scriptbots_app=info");
    env.set("RUST_LOG_STYLE", "never");

    let mut config = ScriptBotsConfig {
        world_width: 200,
        world_height: 140,
        food_cell_size: 20,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 5,
        history_capacity: 640,
        rng_seed: Some(0x51EED5),
        initial_food: 0.3,
        food_max: 1.0,
        food_respawn_interval: 8,
        food_respawn_amount: 0.4,
        food_growth_rate: 0.16,
        food_decay_rate: 0.0008,
        food_diffusion_rate: 0.16,
        reproduction_cooldown: 12,
        reproduction_rate_herbivore: 1.0,
        reproduction_rate_carnivore: 1.0,
        reproduction_energy_cost: 0.2,
        reproduction_child_energy: 0.6,
        reproduction_spawn_jitter: 10.0,
        reproduction_spawn_back_distance: 5.0,
        reproduction_partner_chance: 0.35,
        metabolism_drain: 0.007,
        movement_drain: 0.014,
        temperature_discomfort_rate: 0.001,
        food_intake_rate: 0.009,
        food_waste_rate: 0.0006,
        chart_flush_interval: 240,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 0.5,
        ..ScriptBotsConfig::default()
    };
    config.analytics_stride.behavior_metrics = 24;
    config.analytics_stride.lifecycle_events = 12;
    config.food_sharing_rate = 0.16;
    config.food_transfer_rate = 0.002;

    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
        storage_path
            .to_str()
            .expect("temporary storage path should be utf-8"),
        1,
        1,
        1,
        1,
    )?;
    let analytics = pipeline.analytics_provider();
    let (mut world, persistence) =
        WorldState::with_persistence(config.clone(), Box::new(pipeline.sink()))?;

    let mut rng = SmallRng::seed_from_u64(0xDECAF00D);
    for index in 0..24 {
        let position = Position::new(
            rng.random_range(0.0..config.world_width as f32),
            rng.random_range(0.0..config.world_height as f32),
        );
        let heading = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let color = [
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
        ];
        let health = if index < 4 { -0.01 } else { 1.0 };
        let agent = AgentData::new(
            position,
            Velocity::default(),
            heading,
            health,
            color,
            0.05,
            false,
            0,
            Generation::default(),
        );
        world
            .try_spawn_agent(agent)
            .expect("terminal fixture agent is finite");
    }
    let journal = pipeline.journal_port(HOST_SESSION, Default::default())?;
    let host = terminal_host(world, Some(persistence), Box::new(journal))?;

    let control_config = ControlServerConfig {
        rest_enabled: false,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };

    let (control_runtime, command_submit) = ControlRuntime::launch(host.port(), control_config)?;

    let mut updated_config = config.clone();
    updated_config.food_growth_rate = 0.36;
    updated_config.food_decay_rate = 0.00025;
    updated_config.food_respawn_amount = 0.72;
    updated_config.metabolism_drain = 0.01;
    updated_config.reproduction_cooldown = 10;
    updated_config.reproduction_rate_herbivore = 1.0;
    updated_config.reproduction_rate_carnivore = 1.0;
    updated_config.reproduction_energy_cost = 0.2;
    updated_config.reproduction_child_energy = 0.6;
    updated_config.chart_flush_interval = 90;
    updated_config.reproduction_attempt_interval = 1;
    updated_config.reproduction_attempt_chance = 0.5;
    let submit_ok = command_submit(ControlCommand::UpdateConfig(Box::new(
        updated_config.clone(),
    )));
    assert!(submit_ok.is_some(), "control queue rejected config update");

    let renderer = TerminalRenderer::default();
    {
        let context = RendererContext {
            host: host.port(),
            analytics: analytics.clone(),
            control_runtime: &control_runtime,
            command_submit,
            scenario: Arc::new(ScenarioIdentityV0::caller_seeded("e2e-scenario")),
        };
        renderer.run(context)?;
    }
    control_runtime.shutdown()?;
    let before_finalization = pipeline.flush_and_wait()?;
    let final_owner = host.join()?;
    let cadence = u64::from(config.persistence_interval);
    let cadence_boundary = frames as u64 / cadence * cadence;
    assert_ne!(
        cadence_boundary, frames as u64,
        "fixture must have a partial tail"
    );
    assert_eq!(before_finalization.committed_tick, Some(cadence_boundary));
    assert_eq!(final_owner.required_persistence_tick, Some(frames as u64));
    let shutdown = pipeline.shutdown()?;

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    assert_test_backend_buffer_evidence(&report);
    let summary = &report.summary;

    assert_eq!(summary.frame_count, frames);
    assert_eq!(
        summary.final_tick,
        report.initial.tick + frames as u64,
        "final tick should equal initial tick plus simulated frames"
    );
    assert!(
        summary.total_births > 10,
        "integration run should yield bounded reproduction (births={})",
        summary.total_births
    );
    assert!(
        summary.total_deaths >= 4,
        "integration run should produce observable mortality (deaths={})",
        summary.total_deaths
    );
    assert!(
        summary.final_agent_count > report.initial.agent_count,
        "population should grow over the run (initial={}, final={})",
        report.initial.agent_count,
        summary.final_agent_count
    );
    assert!(
        summary.avg_energy_max > summary.avg_energy_min,
        "mean energy should vary once control updates take effect"
    );

    let total_births: usize = report.frames.iter().map(|frame| frame.births).sum();
    let total_deaths: usize = report.frames.iter().map(|frame| frame.deaths).sum();
    assert_eq!(
        total_births, summary.total_births,
        "frame-wise birth totals should match summary"
    );
    assert_eq!(
        total_deaths, summary.total_deaths,
        "frame-wise death totals should match summary"
    );

    let frames_with_births = report
        .frames
        .iter()
        .filter(|frame| frame.births > 0)
        .count();
    assert!(
        frames_with_births >= 3,
        "birth activity should span many frames (frames_with_births={frames_with_births})"
    );

    let frames_with_deaths = report
        .frames
        .iter()
        .filter(|frame| frame.deaths > 0)
        .count();
    assert!(
        frames_with_deaths >= 1,
        "deaths should appear in multiple frames (frames_with_deaths={frames_with_deaths})"
    );

    let agent_counts: Vec<usize> = report
        .frames
        .iter()
        .map(|frame| frame.agent_count)
        .collect();
    let min_agents = *agent_counts.iter().min().expect("min agent count");
    let max_agents = *agent_counts.iter().max().expect("max agent count");
    assert!(
        max_agents > min_agents,
        "agent count should vary over the run (min={min_agents}, max={max_agents})"
    );
    assert_eq!(
        report.initial.agent_count + summary.total_births - summary.total_deaths,
        summary.final_agent_count,
        "agent conservation should match persistence totals"
    );

    {
        let snapshot = &final_owner.snapshot;
        let world_config = &snapshot.config;
        assert!(
            (world_config.food_growth_rate - updated_config.food_growth_rate).abs() < f32::EPSILON
        );
        assert!(
            (world_config.food_decay_rate - updated_config.food_decay_rate).abs() < f32::EPSILON
        );
        assert!(
            (world_config.metabolism_drain - updated_config.metabolism_drain).abs() < f32::EPSILON
        );
        assert!(
            (world_config.reproduction_rate_herbivore - updated_config.reproduction_rate_herbivore)
                .abs()
                < f32::EPSILON
        );
        assert_eq!(
            world_config.chart_flush_interval, updated_config.chart_flush_interval,
            "chart flush interval should reflect control update"
        );

        let history = &snapshot.summary_history;
        assert!(
            history.len() >= frames,
            "history should capture each simulated tick (len={})",
            history.len()
        );
        let births_in_history = history.iter().filter(|entry| entry.births > 0).count();
        let deaths_in_history = history.iter().filter(|entry| entry.deaths > 0).count();
        assert!(
            births_in_history >= 3,
            "history should record repeated birth activity (birth_ticks={births_in_history})"
        );
        assert!(
            deaths_in_history >= 1,
            "history should record repeated death activity (death_ticks={deaths_in_history})"
        );
    }

    assert!(
        analytics.snapshot().stopped,
        "explicit pipeline shutdown must be visible to frontend readers"
    );
    assert_eq!(shutdown.committed_tick, Some(summary.final_tick));
    assert_eq!(
        shutdown.guarantee,
        scriptbots_storage::PersistenceGuarantee::Durable
    );

    let reader = StorageReader::open(&storage_path.to_string_lossy())?;
    let ledger = reader.run_ledger_summary()?;
    let expected_tick_rows = u64::try_from(frames.div_ceil(5)).expect("frame budget fits in u64");
    assert_eq!(
        ledger.tick_count, expected_tick_rows,
        "storage should contain every cadence boundary plus one final partial batch"
    );

    let latest_tick = ledger
        .latest_tick
        .as_ref()
        .expect("completed run should have a durable tick row");
    assert_eq!(
        latest_tick.tick, summary.final_tick,
        "tick ledger should align with headless summary"
    );
    assert_eq!(
        latest_tick.agent_count, summary.final_agent_count,
        "tick ledger should capture final population size"
    );

    assert_eq!(
        ledger.birth_records,
        u64::try_from(summary.total_births).expect("birth total fits in u64"),
        "born-origin lifecycle records should match the reported reproduction total"
    );

    assert_eq!(
        ledger.death_records,
        u64::try_from(summary.total_deaths).expect("death total fits in u64"),
        "death records should match reported total"
    );

    assert_eq!(
        ledger.birth_events,
        u64::try_from(summary.total_births).expect("birth total fits in u64"),
        "birth events should sum to reported total"
    );

    assert_eq!(
        ledger.death_events,
        u64::try_from(summary.total_deaths).expect("death total fits in u64"),
        "death events should sum to reported total"
    );

    let metrics = reader.recent_metrics(4_096)?;
    let births_metric: f64 = metrics
        .iter()
        .filter(|reading| reading.name == "births.total.count")
        .map(|reading| reading.value)
        .sum();
    assert!(
        (births_metric - summary.total_births as f64).abs() < f64::EPSILON,
        "birth metric samples should sum to reported totals"
    );

    let mortality_metric: f64 = metrics
        .iter()
        .filter(|reading| reading.name == "mortality.total.count")
        .map(|reading| reading.value)
        .sum();
    assert!(
        (mortality_metric - summary.total_deaths as f64).abs() < f64::EPSILON,
        "mortality metric samples should sum to reported totals"
    );

    reader.close()?;
    Ok(())
}
