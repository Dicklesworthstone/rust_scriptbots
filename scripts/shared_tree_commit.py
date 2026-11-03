#!/usr/bin/env python3
"""Commit an explicitly reviewed, immutable snapshot from a shared working tree.

Six agents in this repository share both the working tree and Git's default
index. A normal commit can therefore consume a peer's staged files, while a
pathspec commit can re-read a peer's half-written working-tree bytes after the
caller reviewed an earlier diff.

This tool uses a fresh private index for every review:

    AGENT_NAME=IvoryCondor scripts/shared_tree_commit.py review \
        -m "fix(runtime): retain the command (bd-1234)" \
        -- crates/scriptbots-runtime/src/native.rs

Review the complete diff it prints, then copy its token into:

    AGENT_NAME=IvoryCondor scripts/shared_tree_commit.py commit --review TOKEN

The second command commits exactly the reviewed Git tree, not the then-current
working tree. A short repository mutex serializes HEAD changes, and a
pre-commit plugin rejects commits that did not come through this protocol.

For a conflict-free merge, review with ``--merge-parent COMMIT`` and name every
path changed from HEAD by the merge. The frozen tree must exactly match Git's
clean merge result. Conflicted merges and unrelated in-progress operations
remain unsupported; ordinary reviews still create single-parent commits.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NoReturn, Sequence


PROTOCOL_VERSION = 1
STATE_DIR_NAME = "scriptbots-shared-commit"
HOOK_PLUGIN_NAME = "10-shared-tree-commit"
TOKEN_ENV = "SCRIPTBOTS_SHARED_COMMIT_REVIEW"  # ubs:ignore[py.security.hardcoded-secrets] - environment variable name, not a credential
MANIFEST_ENV = "SCRIPTBOTS_SHARED_COMMIT_MANIFEST"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0


class ProtocolError(RuntimeError):
    """A fail-closed protocol refusal with a stable process exit code."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class Repo:
    root: Path
    common_dir: Path
    state_dir: Path


def fail(message: str, exit_code: int = 1) -> NoReturn:
    raise ProtocolError(message, exit_code)


def run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.run(  # ubs:ignore[python.taint.command] - internal callers choose the executable; user values remain shell-free argv elements
        list(argv),
        cwd=cwd,
        env=env,
        input=input_bytes,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if check and process.returncode != 0:
        stdout = (process.stdout or b"").decode("utf-8", "replace").strip()
        stderr = (process.stderr or b"").decode("utf-8", "replace").strip()
        detail = "\n".join(part for part in (stdout, stderr) if part)
        suffix = f"\n{detail}" if detail else ""
        fail(f"{shlex.join(argv)} failed with exit {process.returncode}{suffix}")
    return process


def git(
    repo: Repo,
    args: Sequence[str],
    *,
    index: Path | None = None,
    check: bool = True,
    capture: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["GIT_LITERAL_PATHSPECS"] = "1"
    if index is None:
        env.pop("GIT_INDEX_FILE", None)
    else:
        env["GIT_INDEX_FILE"] = str(index)
    return run(
        ["git", *args],
        cwd=repo.root,
        env=env,
        check=check,
        capture=capture,
        input_bytes=input_bytes,
    )


def git_text(repo: Repo, args: Sequence[str], *, index: Path | None = None) -> str:
    return git(repo, args, index=index).stdout.decode("utf-8", "strict").strip()


def discover_repo() -> Repo:
    probe = run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        check=True,
    )
    root = Path(probe.stdout.decode("utf-8", "strict").strip()).resolve()
    common_raw = (
        run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root,
            check=True,
        )
        .stdout.decode("utf-8", "strict")
        .strip()
    )
    common = Path(common_raw)
    if not common.is_absolute():
        common = root / common
    common = common.resolve()
    state_dir = common / STATE_DIR_NAME
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return Repo(root=root, common_dir=common, state_dir=state_dir)


def require_agent() -> str:
    agent = os.environ.get("AGENT_NAME", "").strip()
    if not agent:
        fail(
            "AGENT_NAME is required. Export the exact MCP Agent Mail identity "
            "before reviewing or committing.",
            64,
        )
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", agent):
        fail(f"AGENT_NAME has an unsupported shape: {agent!r}", 64)
    return agent


def private_env(index: Path, token: str, manifest_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_LITERAL_PATHSPECS"] = "1"
    env["GIT_INDEX_FILE"] = str(index)
    env[TOKEN_ENV] = token
    env[MANIFEST_ENV] = str(manifest_path)
    return env


def reject_in_progress_git_operation(
    repo: Repo, *, reviewed_merge_parent: str | None = None,
    merge_identity: tuple[int, int] | None = None,
) -> None:
    sentinels = (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-apply",
        "rebase-merge",
    )
    active = []
    for name in sentinels:
        path = git_text(repo, ["rev-parse", "--git-path", name])
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = repo.root / candidate
        if candidate.exists():
            if (
                name == "MERGE_HEAD"
                and reviewed_merge_parent is not None
                and candidate.is_file()
                and not candidate.is_symlink()
                and merge_identity == (candidate.stat().st_dev, candidate.stat().st_ino)
                and candidate.read_bytes() == f"{reviewed_merge_parent}\n".encode()
            ):
                continue
            active.append(name)
    if active:
        fail(
            "shared-tree commits are disabled during an in-progress Git "
            f"operation: {', '.join(active)}"
        )

    unmerged = git(repo, ["diff", "--name-only", "--diff-filter=U", "-z"]).stdout
    if unmerged:
        fail("shared-tree commits are disabled while the default index has conflicts")


def normalize_paths(repo: Repo, raw_paths: Sequence[str]) -> list[str]:
    if not raw_paths:
        fail("at least one explicit repository-relative file path is required", 64)

    normalized: set[str] = set()
    for raw in raw_paths:
        if not raw or "\0" in raw:
            fail(f"invalid empty or NUL-containing path: {raw!r}", 64)
        candidate = Path(raw)
        if candidate.is_absolute():
            fail(f"absolute paths are not accepted: {raw!r}", 64)
        if raw in (".", "./") or any(part == ".." for part in candidate.parts):
            fail(f"broad or parent-relative paths are not accepted: {raw!r}", 64)

        lexical = Path(os.path.normpath(raw))
        if not lexical.parts or lexical.parts[0] == ".git":
            fail(f"Git-internal paths are not committable: {raw!r}", 64)

        absolute = (repo.root / lexical).resolve(strict=False)
        try:
            absolute.relative_to(repo.root)
        except ValueError:
            fail(f"path escapes the repository: {raw!r}", 64)

        relative = lexical.as_posix()
        if absolute.exists() and absolute.is_dir():
            fail(
                f"directories and globs are forbidden; name every file: {relative!r}",
                64,
            )
        if not absolute.exists():
            tracked = git(
                repo,
                ["ls-files", "--error-unmatch", "--", relative],
                check=False,
            )
            if tracked.returncode != 0:
                fail(f"path does not exist and is not tracked: {relative!r}", 64)
        normalized.add(relative)

    return sorted(normalized)

# bd-mdoc: git commit normalizes the message in ways that a naive byte-equal
# comparison misses. This canonical form captures what git actually stores and
# returns through `%B`, so the approved message and the committed message can
# be compared on equal footing. Without it, git's own normalization trips the
# guard on otherwise correct commits.
def canonicalize_commit_message(text: str) -> str:
    """Normalize a commit message to the form git commits and `%B` reads back.

    Git's `commit -m` (and `commit -F`) coalesce trailing whitespace on every
    line, convert CRLF/CR line endings to LF, and drop leading/trailing blank
    lines before storing. `%B` then appends one extra trailing newline that
    callers must rstrip before comparing. Treating any of those differences as
    commit-message corruption produces false positives (bd-mdoc).
    """
    # Git's commit -m only normalizes CRLF -> LF (Unix line-ending canonical),
    # so lone CR stays as CR; mirror that exact conversion.
    text = text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)



@contextmanager
def commit_mutex(
    repo: Repo, timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS
) -> Iterator[None]:
    lock_path = repo.state_dir / "commit.lock"
    descriptor = lock_path.open("a+b")
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    fail(
                        "another reviewed commit holds the repository mutex; "
                        "retry after it finishes",
                        75,
                    )
                time.sleep(0.05)
        yield
    finally:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
        descriptor.close()


def candidate_changed_paths(repo: Repo, base: str, tree: str) -> list[str]:
    raw = git(
        repo,
        [
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            base,
            tree,
            "--",
        ],
    ).stdout
    return sorted(
        item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item
    )


def candidate_diff(repo: Repo, base: str, tree: str) -> bytes:
    return git(
        repo,
        [
            "diff",
            "--binary",
            "--full-index",
            "--no-color",
            "--no-ext-diff",
            "--no-renames",
            base,
            tree,
            "--",
        ],
    ).stdout


def identity_payload(manifest: dict[str, object]) -> dict[str, object]:
    payload = {
        "version": manifest["version"],
        "repository": manifest["repository"],
        "agent": manifest["agent"],
        "base": manifest["base"],
        "tree": manifest["tree"],
        "paths": manifest["paths"],
        "message": manifest["message"],
        "nonce": manifest["nonce"],
    }
    # Keep existing single-parent tokens valid; bind the new parent whenever
    # merge mode is explicitly requested, including during hook validation.
    if "merge_parent" in manifest:
        payload["merge_parent"] = manifest["merge_parent"]
    return payload


def canonical_merge_tree(repo: Repo, base: str, parent: str) -> str:
    resolved = git_text(repo, ["rev-parse", "--verify", f"{parent}^{{commit}}"])
    if resolved != parent or parent == base:
        fail("merge parent must be a distinct pinned commit")
    result = git(repo, ["merge-tree", "--write-tree", base, parent], check=False)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
        fail(f"only conflict-free merges can be reviewed:\n{detail}")
    return result.stdout.decode("ascii", "strict").strip()


def validate_merge_tree(repo: Repo, manifest: dict[str, object]) -> None:
    if "merge_parent" not in manifest:
        return
    parent = manifest["merge_parent"]
    if not isinstance(parent, str):
        fail("review manifest merge_parent has the wrong type")
    expected = canonical_merge_tree(repo, str(manifest["base"]), parent)
    if manifest["tree"] != expected:
        fail("reviewed tree must equal the complete conflict-free merge result")


def merge_head_path(repo: Repo) -> Path:
    path = Path(git_text(repo, ["rev-parse", "--git-path", "MERGE_HEAD"]))
    return path if path.is_absolute() else repo.root / path


def merge_attempt_path(repo: Repo, token: str) -> Path:
    return repo.state_dir / f"merge-attempt-{token}.json"


def validate_merge_attempt(
    repo: Repo, manifest: dict[str, object], token: str
) -> tuple[int, int]:
    path = merge_head_path(repo)
    if not path.is_file() or path.is_symlink():
        fail("reviewed merge is missing its pinned MERGE_HEAD")
    attempt = merge_attempt_path(repo, token)
    if not attempt.is_file() or attempt.is_symlink():
        fail("active MERGE_HEAD is not owned by this review; preserve it and inspect")
    try:
        record = json.loads(attempt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"merge attempt record is unreadable; preserve active state: {error}")
    observed = path.stat()
    identity = (observed.st_dev, observed.st_ino)
    expected = {
        "base": manifest["base"], "parent": manifest["merge_parent"],
        "token": token, "device": identity[0], "inode": identity[1],
    }
    if record != expected or path.read_bytes() != f"{manifest['merge_parent']}\n".encode():
        fail("merge metadata changed; preserve active state and inspect")
    return identity


def token_for(manifest: dict[str, object]) -> str:
    encoded = json.dumps(
        identity_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    if path.exists():
        fail(f"refusing to overwrite an existing review manifest: {path}")
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def load_manifest(repo: Repo, token: str) -> tuple[Path, dict[str, object]]:
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        fail("review token must be the complete 64-character token", 64)
    reviews_dir = (repo.state_dir / "reviews").resolve()
    manifest_path = (reviews_dir / f"{token}.json").resolve()
    try:
        manifest_path.relative_to(reviews_dir)
    except ValueError:
        fail("review manifest escaped the protocol state directory")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail(f"review token is unknown in this clone: {token}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"review manifest is unreadable: {error}")
    if not isinstance(manifest, dict):
        fail("review manifest is not a JSON object")
    return manifest_path, manifest


def validated_index_path(repo: Repo, manifest: dict[str, object]) -> Path:
    raw = manifest.get("index")
    if not isinstance(raw, str):
        fail("review manifest has no private index path")
    index = Path(raw).resolve()
    indexes_dir = (repo.state_dir / "indexes").resolve()
    try:
        index.relative_to(indexes_dir)
    except ValueError:
        fail("private index escaped the protocol state directory")
    if not index.is_file() or index.is_symlink():
        fail(f"private review index is missing or unsafe: {index}")
    return index


def validate_manifest(
    repo: Repo,
    manifest: dict[str, object],
    *,
    expected_token: str,
    require_current_head: bool,
) -> Path:
    required_types: dict[str, type[object]] = {
        "version": int,
        "repository": str,
        "agent": str,
        "base": str,
        "tree": str,
        "paths": list,
        "message": str,
        "nonce": str,
        "diff_sha256": str,
    }
    for key, expected_type in required_types.items():
        if not isinstance(manifest.get(key), expected_type):
            fail(f"review manifest field {key!r} has the wrong type")

    if manifest["version"] != PROTOCOL_VERSION:
        fail(f"unsupported review protocol version: {manifest['version']}")
    if Path(str(manifest["repository"])).resolve() != repo.root:
        fail("review belongs to a different repository")
    if manifest["agent"] != require_agent():
        fail(f"review belongs to another Agent Mail identity: {manifest['agent']!r}")
    if token_for(manifest) != expected_token:
        fail("review token does not match its manifest")

    paths_value = manifest["paths"]
    if (
        not isinstance(paths_value, list)
        or not paths_value
        or not all(isinstance(path, str) for path in paths_value)
        or paths_value != sorted(set(paths_value))
    ):
        fail("review manifest paths are empty, duplicated, or non-canonical")

    base = str(manifest["base"])
    tree = str(manifest["tree"])
    if require_current_head:
        current_head = git_text(repo, ["rev-parse", "HEAD"])
        if current_head != base:
            fail(
                "HEAD moved after review; the immutable candidate is stale. "
                f"Reviewed {base[:12]}, current {current_head[:12]}. Run review again.",
                75,
            )

    index = validated_index_path(repo, manifest)
    actual_tree = git_text(repo, ["write-tree"], index=index)
    if actual_tree != tree:
        fail(
            "private review index changed after approval: "
            f"expected tree {tree}, found {actual_tree}"
        )

    actual_paths = candidate_changed_paths(repo, base, tree)
    if actual_paths != paths_value:
        fail(
            "candidate path set no longer matches approval: "
            f"expected {paths_value!r}, found {actual_paths!r}"
        )

    diff = candidate_diff(repo, base, tree)
    actual_diff_hash = hashlib.sha256(diff).hexdigest()
    if actual_diff_hash != manifest["diff_sha256"]:
        fail("candidate diff bytes no longer match the reviewed diff")
    validate_merge_tree(repo, manifest)
    return index


def review_candidate(args: argparse.Namespace) -> int:
    repo = discover_repo()
    agent = require_agent()
    reject_in_progress_git_operation(repo)
    paths = normalize_paths(repo, args.paths)
    # bd-mdoc: canonicalize so trailing whitespace, CRLF/CR line endings, and
    # leading/trailing blank lines are normalized to the form git actually
    # stores; otherwise the byte-equal comparison at commit time fires on
    # git's own normalization (see canonicalize_commit_message).
    message = canonicalize_commit_message(args.message)
    if not message:
        fail("commit message must not be empty", 64)

    indexes_dir = repo.state_dir / "indexes"
    reviews_dir = repo.state_dir / "reviews"
    indexes_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    reviews_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    nonce = f"{time.time_ns():x}-{os.getpid():x}"
    safe_agent = re.sub(r"[^A-Za-z0-9_.-]", "_", agent)
    index = (indexes_dir / f"{safe_agent}-{nonce}.index").resolve()

    with commit_mutex(repo, args.lock_timeout):
        reject_in_progress_git_operation(repo)
        base = git_text(repo, ["rev-parse", "HEAD"])
        merge_parent = (
            git_text(repo, ["rev-parse", "--verify", f"{args.merge_parent}^{{commit}}"])
            if args.merge_parent else None
        )
        git(repo, ["read-tree", base], index=index)
        index.chmod(0o600)
        git(repo, ["add", "-A", "--", *paths], index=index)
        tree = git_text(repo, ["write-tree"], index=index)
        actual_paths = candidate_changed_paths(repo, base, tree)
        if actual_paths != paths:
            fail(
                "every declared path must change, and no undeclared path may change. "
                f"Declared {paths!r}; candidate changed {actual_paths!r}."
            )
        whitespace = git(
            repo,
            ["diff", "--check", "--no-renames", base, tree, "--"],
            check=False,
        )
        if whitespace.returncode != 0:
            detail = (whitespace.stdout + whitespace.stderr).decode("utf-8", "replace")
            fail(f"candidate fails git diff --check:\n{detail.rstrip()}")
        if git_text(repo, ["rev-parse", "HEAD"]) != base:
            fail("HEAD moved while the review snapshot was being created; retry", 75)

    diff = candidate_diff(repo, base, tree)
    manifest: dict[str, object] = {
        "version": PROTOCOL_VERSION,
        "repository": str(repo.root),
        "agent": agent,
        "base": base,
        "tree": tree,
        "paths": paths,
        "message": message,
        "nonce": nonce,
        "index": str(index),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "created_unix_ns": time.time_ns(),
    }
    if merge_parent is not None:
        manifest["merge_parent"] = merge_parent
        validate_merge_tree(repo, manifest)
    token = token_for(manifest)
    manifest["token"] = token
    manifest_path = reviews_dir / f"{token}.json"
    write_manifest(manifest_path, manifest)

    staged_elsewhere = git(
        repo, ["diff", "--cached", "--name-only", "-z"], check=True
    ).stdout
    staged_count = len([part for part in staged_elsewhere.split(b"\0") if part])

    print("shared-tree-commit: immutable candidate ready")
    print(f"  agent:   {agent}")
    print(f"  base:    {base}")
    if merge_parent is not None:
        print(f"  merge:   {merge_parent}")
    print(f"  tree:    {tree}")
    print(f"  message: {message}")
    print("  paths:")
    for path in paths:
        print(f"    {path!r}")
    if staged_count:
        print(
            f"  note: ignored and will preserve {staged_count} path(s) staged "
            "in the shared index"
        )
    print("\n===== BEGIN EXACT REVIEW DIFF =====")
    sys.stdout.flush()
    sys.stdout.buffer.write(diff)
    if diff and not diff.endswith(b"\n"):
        sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    print("===== END EXACT REVIEW DIFF =====")
    print(f"\nREVIEW TOKEN: {token}")
    print("After reviewing every hunk, commit exactly this frozen tree with:")
    print(
        f"  AGENT_NAME={shlex.quote(agent)} "
        f"scripts/shared_tree_commit.py commit --review {token}"
    )
    return 0


def reconcile_default_index(repo: Repo, paths: Sequence[str], new_head: str) -> None:
    result = git(
        repo,
        ["reset", "--quiet", new_head, "--", *paths],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
        fail(
            "commit landed, but the shared index could not be reconciled for "
            f"the committed paths. Do not reset or clean anything. Commit: {new_head}"
            + (f"\n{detail}" if detail else "")
        )


def commit_review(args: argparse.Namespace) -> int:
    repo = discover_repo()
    manifest_path, manifest = load_manifest(repo, args.review)

    with commit_mutex(repo, args.lock_timeout):
        index = validate_manifest(
            repo,
            manifest,
            expected_token=args.review,
            require_current_head=True,
        )
        base = str(manifest["base"])
        tree = str(manifest["tree"])
        paths = list(manifest["paths"])
        message = str(manifest["message"])

        env = private_env(index, args.review, manifest_path)
        merge_parent = manifest.get("merge_parent")
        merge_path = merge_head_path(repo)
        merge_identity = None
        if merge_parent is not None and merge_path.exists():
            merge_identity = validate_merge_attempt(repo, manifest, args.review)
        reject_in_progress_git_operation(
            repo, reviewed_merge_parent=merge_parent, merge_identity=merge_identity,
        )
        if merge_parent is not None:
            if merge_identity is None:
                # Never remove or rename active Git metadata. A failed hook
                # leaves this exact file for a token-bound retry; partial writes
                # or lost/replaced state fail closed for manual inspection.
                attempt_path = merge_attempt_path(repo, args.review)
                if attempt_path.exists():
                    fail("merge attempt lost its active state; preserve the record and inspect")
                with merge_path.open("xb") as handle:
                    observed = os.fstat(handle.fileno())
                    write_manifest(attempt_path, {
                        "base": base, "parent": merge_parent, "token": args.review,
                        "device": observed.st_dev, "inode": observed.st_ino,
                    })
                    handle.write(f"{merge_parent}\n".encode())
                    handle.flush()
                    os.fsync(handle.fileno())
        process = run(
            ["git", "commit", "-m", message],
            cwd=repo.root,
            env=env,
            check=False,
            capture=False,
        )
        if process.returncode != 0:
            fail(
                "Git refused the reviewed commit. The frozen candidate remains "
                "available for retry while HEAD is unchanged.",
                process.returncode,
            )

        new_head = git_text(repo, ["rev-parse", "HEAD"])
        parent_line = git_text(repo, ["rev-list", "--parents", "-n", "1", new_head])
        parent_parts = parent_line.split()
        expected_parents = [new_head, base]
        if merge_parent is not None:
            expected_parents.append(str(merge_parent))
        if parent_parts != expected_parents:
            fail(
                "commit landed with an unexpected parent set; stop and inspect "
                f"{new_head}. Expected parents {expected_parents[1:]}."
            )
        committed_tree = git_text(repo, ["show", "-s", "--format=%T", new_head])
        if committed_tree != tree:
            fail(
                "commit landed with a tree different from the approved candidate; "
                f"stop and inspect {new_head}."
            )
        committed_paths = candidate_changed_paths(repo, base, committed_tree)
        if committed_paths != paths:
            fail(
                "commit landed with a path set different from approval; "
                f"stop and inspect {new_head}."
            )
        committed_message = git(repo, ["show", "-s", "--format=%B", new_head]).stdout
        # bd-mdoc: compare canonical forms so git's own normalization of
        # line endings, trailing whitespace, and trailing blank lines does
        # not trip the guard on otherwise correct commits.
        if (
            canonicalize_commit_message(committed_message.decode("utf-8", "strict"))
            != message
        ):
            fail(
                "commit landed with a message different from approval; "
                f"stop and inspect {new_head}."
            )

        reconcile_default_index(repo, paths, new_head)

    print(f"shared-tree-commit: committed reviewed tree {tree[:12]} as {new_head[:12]}")
    return 0


def hook_check() -> int:
    repo = discover_repo()
    token = os.environ.get(TOKEN_ENV, "").strip()
    manifest_env = os.environ.get(MANIFEST_ENV, "").strip()
    if not token or not manifest_env:
        fail(
            "raw commits are disabled in this shared working tree.\n"
            "Run `scripts/shared_tree_commit.py review -m MESSAGE -- FILE...`, "
            "review every hunk, then run its tokenized commit command."
        )

    manifest_path, manifest = load_manifest(repo, token)
    if Path(manifest_env).resolve() != manifest_path:
        fail("commit environment names a different review manifest")
    index = validate_manifest(
        repo,
        manifest,
        expected_token=token,
        require_current_head=True,
    )
    reject_in_progress_git_operation(
        repo, reviewed_merge_parent=manifest.get("merge_parent"),
        merge_identity=(
            validate_merge_attempt(repo, manifest, token)
            if "merge_parent" in manifest else None
        ),
    )
    current_index = os.environ.get("GIT_INDEX_FILE", "").strip()
    if not current_index or Path(current_index).resolve() != index:
        fail("Git is not committing the private index bound to this review")
    print(
        "shared-tree-commit: approved immutable tree "
        f"{manifest['tree']} ({len(manifest['paths'])} path(s))"
    )
    return 0


def install_hook() -> int:
    repo = discover_repo()
    chain_dir = repo.common_dir / "hooks" / "hooks.d" / "pre-commit"
    chain_runner = repo.common_dir / "hooks" / "pre-commit"
    if not chain_runner.is_file() or not chain_dir.is_dir():
        fail(
            "the Agent Mail pre-commit chain runner is not installed; "
            "refusing to replace or invent the repository hook chain"
        )
    plugin = chain_dir / HOOK_PLUGIN_NAME
    content = (
        "#!/usr/bin/env bash\n"
        "# Installed by scripts/shared_tree_commit.py install-hook (bd-qsi4).\n"
        'exec "$(git rev-parse --show-toplevel)/scripts/shared_tree_commit.py" hook\n'
    )
    if plugin.exists():
        if not plugin.is_file() or plugin.is_symlink():
            fail(f"refusing to replace unsafe existing hook plugin: {plugin}")
        if plugin.read_text(encoding="utf-8") != content:
            fail(f"refusing to overwrite a different existing hook plugin: {plugin}")
    else:
        with plugin.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    plugin.chmod(0o755)
    print(f"shared-tree-commit: installed {plugin}")
    return 0


def fixture_git(
    root: Path, args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return run(["git", *args], cwd=root, check=check)


def fixture_write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def self_test() -> int:
    """Exercise the incident classes in a retained disposable repository."""

    script = Path(__file__).resolve()
    fixture = Path(tempfile.mkdtemp(prefix="bd-qsi4-shared-tree-commit-"))
    print(f"self-test fixture retained at {fixture}")
    fixture_git(fixture, ["init", "-b", "main"])
    fixture_git(fixture, ["config", "user.name", "Shared Tree Test"])
    fixture_git(fixture, ["config", "user.email", "shared-tree@example.invalid"])
    fixture_write(fixture, "a.txt", "A0\n")
    fixture_write(fixture, "b.txt", "B0\n")
    fixture_write(fixture, "c.txt", "C0\n")
    fixture_git(fixture, ["add", "a.txt", "b.txt", "c.txt"])
    fixture_git(fixture, ["commit", "-m", "base"])

    hook = fixture / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} hook || exit $?\n"
        'git diff --cached --name-only -z >"$PWD/.git/hook-seen"\n'
        'test ! -f "$PWD/.git/reject-reviewed-merge"\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)
    env = os.environ.copy()
    env["AGENT_NAME"] = "TestFalcon"

    def protocol(
        arguments: Sequence[str], *, expect_ok: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        result = run(
            [sys.executable, str(script), *arguments],
            cwd=fixture,
            env=env,
            check=False,
        )
        if expect_ok and result.returncode != 0:
            detail = (result.stdout + result.stderr).decode("utf-8", "replace")
            fail(f"self-test protocol command failed:\n{detail}")
        if not expect_ok and result.returncode == 0:
            fail("self-test expected protocol refusal, but command succeeded")
        return result

    def reviewed_token(
        paths: str | Sequence[str], message: str, *, merge_parent: str | None = None
    ) -> str:
        arguments = ["review", "-m", message]
        if merge_parent is not None:
            arguments.extend(["--merge-parent", merge_parent])
        arguments.extend(["--", *([paths] if isinstance(paths, str) else paths)])
        result = protocol(arguments)
        output = result.stdout.decode("utf-8", "replace")
        match = re.search(r"^REVIEW TOKEN: ([0-9a-f]{64})$", output, re.MULTILINE)
        if not match:
            fail(f"self-test could not parse review token:\n{output}")
        return match.group(1)

    # Reproduce both incident families at once. Shared-index B must neither be
    # swept nor lost, and post-review working-tree A must not replace frozen A.
    fixture_write(fixture, "b.txt", "B-peer-staged\n")
    fixture_git(fixture, ["add", "b.txt"])
    fixture_write(fixture, "a.txt", "A-reviewed\n")
    token = reviewed_token("a.txt", "test: immutable A")
    fixture_write(fixture, "a.txt", "A-later-unreviewed\n")
    protocol(["commit", "--review", token])

    head_a = fixture_git(fixture, ["show", "HEAD:a.txt"]).stdout
    head_b = fixture_git(fixture, ["show", "HEAD:b.txt"]).stdout
    worktree_a = (fixture / "a.txt").read_bytes()
    staged = fixture_git(fixture, ["diff", "--cached", "--name-only", "-z"]).stdout
    hook_seen = (fixture / ".git" / "hook-seen").read_bytes()
    if head_a != b"A-reviewed\n":
        fail("self-test: commit did not preserve the reviewed A snapshot")
    if worktree_a != b"A-later-unreviewed\n":
        fail("self-test: later working-tree A edit was lost")
    if head_b != b"B0\n" or staged != b"b.txt\0":
        fail("self-test: peer-staged B was swept or lost")
    if hook_seen != b"a.txt\0":
        fail("self-test: pre-commit hook did not inspect the private candidate index")
    print("  ok: private candidate freezes bytes and preserves peer staging")

    # A direct/raw commit must fail before it can consume the shared index.
    before_raw = fixture_git(fixture, ["rev-parse", "HEAD"]).stdout
    raw = run(
        ["git", "commit", "-m", "raw commit must fail"],
        cwd=fixture,
        env=env,
        check=False,
    )
    after_raw = fixture_git(fixture, ["rev-parse", "HEAD"]).stdout
    if raw.returncode == 0 or before_raw != after_raw:
        fail("self-test: raw commit was not blocked")
    print("  ok: raw shared-index commit is blocked")

    # Two candidates reviewed at one base cannot both commit. The first advances
    # HEAD; the second must fail stale rather than reverting the first.
    fixture_git(fixture, ["reset", "--quiet", "HEAD", "--", "b.txt"])
    fixture_write(fixture, "a.txt", "A-second\n")
    token_a = reviewed_token("a.txt", "test: second A")
    fixture_write(fixture, "c.txt", "C-second\n")
    token_c = reviewed_token("c.txt", "test: second C")
    protocol(["commit", "--review", token_a])
    stale = protocol(["commit", "--review", token_c], expect_ok=False)
    stale_text = (stale.stdout + stale.stderr).decode("utf-8", "replace")
    if "HEAD moved after review" not in stale_text:
        fail(f"self-test: stale candidate failed for the wrong reason:\n{stale_text}")
    if fixture_git(fixture, ["show", "HEAD:a.txt"]).stdout != b"A-second\n":
        fail("self-test: stale candidate disturbed the first committed tree")
    print("  ok: HEAD compare-and-swap rejects a stale private index")

    # Tampering with the frozen private index must invalidate approval.
    fixture_write(fixture, "c.txt", "C-reviewed\n")
    token_tamper = reviewed_token("c.txt", "test: tamper refusal")
    repo = Repo(
        root=fixture,
        common_dir=fixture / ".git",
        state_dir=fixture / ".git" / STATE_DIR_NAME,
    )
    _, manifest = load_manifest(repo, token_tamper)
    index = validated_index_path(repo, manifest)
    fixture_write(fixture, "c.txt", "C-tampered\n")
    git(repo, ["add", "-A", "--", "c.txt"], index=index)
    tampered = protocol(["commit", "--review", token_tamper], expect_ok=False)
    tampered_text = (tampered.stdout + tampered.stderr).decode("utf-8", "replace")
    if "private review index changed" not in tampered_text:
        fail(
            "self-test: modified candidate failed for the wrong reason:\n"
            f"{tampered_text}"
        )
    print("  ok: candidate tree tampering invalidates approval")

    # Construct an actual divergent peer history. Clone hooks are intentionally
    # absent in this isolated peer; the repository under test retains its hook.
    peer = Path(tempfile.mkdtemp(prefix="bd-h4n2-merge-peer-"))
    fixture_git(fixture, ["clone", "--local", str(fixture), str(peer)])
    fixture_git(peer, ["config", "user.name", "Merge Peer"])
    fixture_git(peer, ["config", "user.email", "merge-peer@example.invalid"])
    fixture_write(peer, "a.txt", "A-from-peer\n")
    fixture_write(peer, "d.txt", "D-from-peer\n")
    fixture_git(peer, ["add", "a.txt", "d.txt"])
    fixture_git(peer, ["commit", "-m", "peer A and D"])
    fixture_git(fixture, ["fetch", str(peer), "main"])
    parent = fixture_git(fixture, ["rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
    fixture_write(fixture, "c.txt", "C-local-divergence\n")
    protocol(["commit", "--review", reviewed_token("c.txt", "test: local divergence")])

    fixture_write(fixture, "a.txt", "A-from-peer\n")
    partial = protocol(
        ["review", "--merge-parent", parent, "-m", "incomplete merge", "--", "a.txt"],
        expect_ok=False,
    )
    if b"complete conflict-free merge result" not in partial.stderr:
        fail("self-test: omitted peer path was not rejected as an incomplete merge")
    print("  ok: merge review rejects an omitted incoming path")

    fixture_write(fixture, "d.txt", "D-from-peer\n")
    token_merge = reviewed_token(["a.txt", "d.txt"], "test: merge", merge_parent=parent)
    merge_manifest_path, merge_manifest = load_manifest(repo, token_merge)
    original_manifest = merge_manifest_path.read_bytes()
    changed_parent = dict(merge_manifest)
    changed_parent["merge_parent"] = str(merge_manifest["base"])
    merge_manifest_path.write_text(json.dumps(changed_parent), encoding="utf-8")
    wrong_parent = protocol(["commit", "--review", token_merge], expect_ok=False)
    merge_manifest_path.write_bytes(original_manifest)
    if b"token does not match" not in wrong_parent.stderr:
        fail("self-test: changing the approved second parent did not invalidate its token")
    print("  ok: merge token binds the exact second parent")

    merge_index = validated_index_path(repo, merge_manifest)
    # private_env inherits this process's identity, whereas protocol() sets the
    # fixture identity explicitly. Bind it before exercising the hook directly.
    hook_env = private_env(merge_index, token_merge, merge_manifest_path)
    hook_env["AGENT_NAME"] = "TestFalcon"
    missing_head = run(
        [sys.executable, str(script), "hook"], cwd=fixture, env=hook_env, check=False,
    )
    if missing_head.returncode == 0 or b"missing its pinned MERGE_HEAD" not in missing_head.stderr:
        fail("self-test: merge hook accepted a missing second-parent sentinel")

    merge_head = fixture / ".git" / "MERGE_HEAD"
    merge_head.write_text(f"{merge_manifest['base']}\n", encoding="ascii")
    foreign = protocol(["commit", "--review", token_merge], expect_ok=False)
    foreign_hook = run(
        [sys.executable, str(script), "hook"], cwd=fixture, env=hook_env, check=False,
    )
    if (
        b"not owned by this review" not in foreign.stderr
        or foreign_hook.returncode == 0
        or b"not owned by this review" not in foreign_hook.stderr
        or merge_head.read_text() != f"{merge_manifest['base']}\n"
    ):
        fail("self-test: foreign merge state was accepted or disturbed")
    merge_head.rename(fixture / ".git" / "foreign-merge-retained")
    print("  ok: merge hook and command reject missing/foreign merge state")

    fixture_write(fixture, "c.txt", "C-advanced-before-merge\n")
    protocol(["commit", "--review", reviewed_token("c.txt", "test: advance merge base")])
    stale_merge = protocol(["commit", "--review", token_merge], expect_ok=False)
    if b"HEAD moved after review" not in stale_merge.stderr:
        fail("self-test: stale merge candidate was not rejected")
    print("  ok: merge review cannot overwrite an advanced local base")

    token_merge = reviewed_token(["a.txt", "d.txt"], "test: merge", merge_parent=parent)
    before_merge = fixture_git(fixture, ["rev-parse", "HEAD"]).stdout.decode().strip()
    expected_tree = canonical_merge_tree(repo, before_merge, parent)
    fixture_write(fixture, "b.txt", "B-peer-staged-again\n")
    fixture_git(fixture, ["add", "b.txt"])
    fixture_write(fixture, "c.txt", "C-next-unreviewed-edit\n")
    normal_pending = reviewed_token("c.txt", "test: normal commit blocked during merge")
    fixture_write(fixture, "a.txt", "A-later-unreviewed-merge-edit\n")
    reject_flag = fixture / ".git" / "reject-reviewed-merge"
    reject_flag.write_text("exercise the real subsequent hook\n", encoding="ascii")
    rejected = protocol(["commit", "--review", token_merge], expect_ok=False)
    if (
        b"Git refused the reviewed commit" not in rejected.stderr
        or fixture_git(fixture, ["rev-parse", "HEAD"]).stdout.decode().strip() != before_merge
        or merge_head.read_bytes() != f"{parent}\n".encode()
        or not merge_attempt_path(repo, token_merge).is_file()
    ):
        fail("self-test: rejected merge did not retain its candidate and owned merge state")
    normal_during_merge = protocol(["commit", "--review", normal_pending], expect_ok=False)
    if b"in-progress Git operation" not in normal_during_merge.stderr:
        fail("self-test: a normal token was accepted while a reviewed merge is pending")
    owned_head = fixture / ".git" / "owned-merge-head-retained"
    merge_head.rename(owned_head)
    merge_head.write_text(f"{parent}\n", encoding="ascii")
    replaced = protocol(["commit", "--review", token_merge], expect_ok=False)
    if b"merge metadata changed" not in replaced.stderr:
        fail("self-test: same-content replacement MERGE_HEAD reused another file's ownership")
    merge_head.rename(fixture / ".git" / "replaced-merge-head-retained")
    owned_head.rename(merge_head)
    reject_flag.rename(fixture / ".git" / "rejected-merge-control-retained")
    protocol(["commit", "--review", token_merge])
    merged = fixture_git(fixture, ["rev-parse", "HEAD"]).stdout.decode().strip()
    if (
        fixture_git(fixture, ["rev-list", "--parents", "-n", "1", merged]).stdout.decode().split()
        != [merged, before_merge, parent]
        or fixture_git(fixture, ["rev-parse", "HEAD^{tree}"]).stdout.decode().strip() != expected_tree
        or fixture_git(fixture, ["show", "HEAD:a.txt"]).stdout != b"A-from-peer\n"
        or fixture_git(fixture, ["show", "HEAD:c.txt"]).stdout != b"C-advanced-before-merge\n"
        or fixture_git(fixture, ["show", "HEAD:d.txt"]).stdout != b"D-from-peer\n"
        or fixture_git(fixture, ["show", "HEAD:b.txt"]).stdout != b"B0\n"
        or fixture_git(fixture, ["diff", "--cached", "--name-only", "-z"]).stdout != b"b.txt\0"
        or fixture_git(fixture, ["show", ":b.txt"]).stdout != b"B-peer-staged-again\n"
        or (fixture / "a.txt").read_bytes() != b"A-later-unreviewed-merge-edit\n"
        or (fixture / "c.txt").read_bytes() != b"C-next-unreviewed-edit\n"
        or (fixture / ".git" / "hook-seen").read_bytes() != b"a.txt\0d.txt\0"
        or merge_head.exists()
    ):
        fail("self-test: merge lost its canonical tree, exact parents, hook, or peer edits")
    print("  ok: real two-parent merge runs hooks, preserves peer staging/edits, and retries safely")
    print("  ok: pending merge blocks normal commits and refuses replacement metadata")

    fixture_write(peer, "c.txt", "C-peer-conflict\n")
    fixture_git(peer, ["add", "c.txt"])
    fixture_git(peer, ["commit", "-m", "peer conflicts with local C"])
    fixture_git(fixture, ["fetch", str(peer), "main"])
    conflicting_parent = fixture_git(fixture, ["rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
    conflict = protocol(
        ["review", "--merge-parent", conflicting_parent, "-m", "conflicted merge", "--", "a.txt"],
        expect_ok=False,
    )
    if b"only conflict-free merges can be reviewed" not in conflict.stderr:
        fail("self-test: conflicted history was not refused by the canonical merge calculation")
    if fixture_git(fixture, ["rev-parse", "HEAD"]).stdout.decode().strip() != merged or merge_head.exists():
        fail("self-test: conflict refusal disturbed HEAD or created active merge state")
    print("  ok: conflicted history is rejected without changing HEAD or active Git state")
    print(f"merge peer fixture retained at {peer}")

    print("self-test: all shared-tree commit invariants passed")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        description="Review and commit immutable snapshots in a shared Git working tree."
    )
    subcommands = top.add_subparsers(dest="command", required=True)

    review = subcommands.add_parser(
        "review", help="freeze and print an exact candidate diff"
    )
    review.add_argument("-m", "--message", required=True)
    review.add_argument(
        "--merge-parent",
        help="pin a second parent and require Git's exact conflict-free merge tree",
    )
    review.add_argument(
        "--lock-timeout",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SECONDS,
        help=argparse.SUPPRESS,
    )
    review.add_argument("paths", nargs="+")
    review.set_defaults(handler=review_candidate)

    commit = subcommands.add_parser(
        "commit", help="commit an explicitly approved review token"
    )
    commit.add_argument("--review", required=True)
    commit.add_argument(
        "--lock-timeout",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SECONDS,
        help=argparse.SUPPRESS,
    )
    commit.set_defaults(handler=commit_review)

    hook = subcommands.add_parser("hook", help=argparse.SUPPRESS)
    hook.set_defaults(handler=lambda _args: hook_check())

    install = subcommands.add_parser(
        "install-hook",
        help="install the enforcement plugin into Agent Mail's hook chain",
    )
    install.set_defaults(handler=lambda _args: install_hook())

    tests = subcommands.add_parser(
        "self-test", help="run retained disposable-repository acceptance tests"
    )
    tests.set_defaults(handler=lambda _args: self_test())
    return top


def main() -> int:
    try:
        args = parser().parse_args()
        if getattr(args, "lock_timeout", 0) < 0:
            fail("--lock-timeout must be non-negative", 64)
        return int(args.handler(args))
    except ProtocolError as error:
        print(f"shared-tree-commit: REFUSED: {error}", file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print("shared-tree-commit: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
