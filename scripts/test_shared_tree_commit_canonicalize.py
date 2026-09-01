#!/usr/bin/env python3
"""Regression tests for scripts/shared_tree_commit.py.

bd-mdoc: the byte-equal comparison between the approved commit message and
what git stores/reads back can disagree purely because of git's own
normalization (CRLF -> LF, trailing whitespace, leading/trailing blank lines).
`canonicalize_commit_message` must collapse those differences; these tests
pin the contract.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "shared_tree_commit.py"

def load_module():
    spec = importlib.util.spec_from_file_location("shared_tree_commit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["shared_tree_commit"] = module  # register before exec so @dataclass can introspect
    spec.loader.exec_module(module)
    return module



class CanonicalizeCommitMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module()
        cls.canonicalize = staticmethod(cls.mod.canonicalize_commit_message)

    def test_simple(self):
        self.assertEqual(self.canonicalize("sub\n\nbody"), "sub\n\nbody")

    def test_trailing_newline_stripped(self):
        # `git commit -m` drops trailing newlines.
        self.assertEqual(self.canonicalize("sub\n\nbody\n"), "sub\n\nbody")

    def test_trailing_blank_line_stripped(self):
        # Two trailing newlines (one blank line at end) collapse.
        self.assertEqual(self.canonicalize("sub\n\nbody\n\n"), "sub\n\nbody")

    def test_many_trailing_newlines_stripped(self):
        self.assertEqual(
            self.canonicalize("sub\n\nbody\n\n\n\n\n"), "sub\n\nbody"
        )

    def test_leading_blank_lines_stripped(self):
        self.assertEqual(
            self.canonicalize("\n\nsub\n\nbody\n\n"), "sub\n\nbody"
        )

    def test_trailing_whitespace_per_line_stripped(self):
        self.assertEqual(
            self.canonicalize("sub\n\nbody  \t"), "sub\n\nbody"
        )

    def test_crlf_normalized_to_lf(self):
        # Git's commit -m converts CRLF to LF.
        self.assertEqual(self.canonicalize("sub\r\n\r\nbody"), "sub\n\nbody")

    def test_lone_cr_preserved(self):
        # Git only normalizes CRLF -> LF; lone CR stays.
        self.assertEqual(self.canonicalize("sub\rbody"), "sub\rbody")

    def test_single_line(self):
        self.assertEqual(self.canonicalize("single"), "single")

    def test_empty_string(self):
        self.assertEqual(self.canonicalize(""), "")

    def test_whitespace_only(self):
        # All-whitespace input yields empty.
        self.assertEqual(self.canonicalize("   \n\n  \t\n"), "")

    def test_internal_blank_lines_preserved(self):
        # Blank lines between paragraphs must NOT collapse; they are part of
        # the canonical form (one blank line between subject and body).
        self.assertEqual(
            self.canonicalize("sub\n\n\nbody"), "sub\n\n\nbody"
        )

    def test_bd_mdoc_regression(self):
        # Reproduces commit 3e16ac07a's exact pattern: approved message
        # contains one blank line between subject and body (the user's
        # newline convention differs by a single trailing whitespace
        # or blank line); canonicalization must make the comparison
        # report equal.
        approved = "feat(storage): wire the interactions table (bd-2z0.5.9)"
        approved_body = (
            "bd-2z0.5.9 asks for a new interactions table.\n"
            "\n"
            "This wires it."
        )
        full_approved = approved + "\n\n" + approved_body + "\n"
        # What %B returns for the same content stored via `git commit -m`:
        stored_with_trailing_blank = full_approved + "\n"
        # Both should canonicalize identically.
        self.assertEqual(
            self.canonicalize(full_approved),
            self.canonicalize(stored_with_trailing_blank),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)