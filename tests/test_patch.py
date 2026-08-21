"""Tests for the strict Formal Disco diff applier (extender step 2).

Format: @@ anchors, = keep, - delete, + add — semantics identical to
formal-disco's apply_text_diff, with one deliberate difference: a
directive that fails to match RAISES PatchError instead of being
silently skipped. The silent skip is what let the Stage-2 Gemini Fixer
produce half-applied garbage; here a malformed patch fails cleanly and
the caller keeps the original text.

The three hand-written patches required by the step are the fixtures:
PATCH_ADD_REGISTER, PATCH_MODIFY_ALWAYS, PATCH_MALFORMED, written
against the real g0_014 token_bucket design from the corpus.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extender.patch import PatchError, apply_patch   # noqa: E402


# --- format compatibility: formal-disco's own example, verbatim ---

FD_BEFORE = "hello\nworld\na\nb cd\nline2\nline3\n"
FD_DIFF = """\
@@ a @@
+ inserted-after-a
@@ b cd @@
- line2
= line3
+ after3"""
FD_AFTER = "hello\nworld\na\ninserted-after-a\nb cd\nline3\nafter3\n"


def test_formal_disco_example_applies_identically():
    assert apply_patch(FD_BEFORE, FD_DIFF) == FD_AFTER


# --- the real design the hand patches target: g0_014 token_bucket ---

TOKEN_BUCKET = """\
module token_bucket (
    input wire clk,
    input wire rst,
    input wire drip,
    input wire spend,
    output reg [2:0] tokens,
    output reg tokens_hi
);
    wire drip_en  = drip  && (tokens != 3'd4);
    wire spend_en = spend && (tokens != 3'd0);
    wire inc = drip_en && !spend_en;
    wire dec = spend_en && !drip_en;

    initial tokens   = 3'd0;
    initial tokens_hi = 1'b0;

    always @(posedge clk) begin
        if (rst) begin
            tokens    <= 3'd0;
            tokens_hi <= 1'b0;
        end else begin
            if (inc)      tokens <= tokens + 3'd1;
            else if (dec) tokens <= tokens - 3'd1;

            if (inc && tokens == 3'd1)      tokens_hi <= 1'b1;
            else if (dec && tokens == 3'd2) tokens_hi <= 1'b0;
        end
    end

    always @(posedge clk)
        if (tokens_hi) assert (tokens != 3'd0);
endmodule
"""

# Hand patch 1: add a register — a spend counter with declaration,
# reset, and update, threaded through three anchor points.
PATCH_ADD_REGISTER = """\
@@ output reg tokens_hi @@
@@ ); @@
+     reg [3:0] spent_total;
@@ initial tokens_hi = 1'b0; @@
+     initial spent_total = 4'd0;
@@ if (rst) begin @@
@@ tokens    <= 3'd0; @@
@@ tokens_hi <= 1'b0; @@
+             spent_total <= 4'd0;
@@ end else begin @@
+             if (dec) spent_total <= spent_total + 4'd1;"""

# Hand patch 2: modify a line inside the always block — the tokens_hi
# set threshold becomes >= instead of ==.
PATCH_MODIFY_ALWAYS = """\
@@ end else begin @@
- if (inc && tokens == 3'd1)      tokens_hi <= 1'b1;
+             if (inc && tokens >= 3'd1)      tokens_hi <= 1'b1;"""

# Hand patch 3: deliberately malformed — the anchor exists nowhere.
PATCH_MALFORMED = """\
@@ this line exists nowhere in the module @@
+     reg ghost;"""


def test_add_register_patch_applies():
    out = apply_patch(TOKEN_BUCKET, PATCH_ADD_REGISTER)
    assert "reg [3:0] spent_total;" in out
    assert "initial spent_total = 4'd0;" in out
    assert "spent_total <= spent_total + 4'd1;" in out
    # original logic untouched
    assert "if (tokens_hi) assert (tokens != 3'd0);" in out


def test_add_register_patch_places_lines_correctly():
    out = apply_patch(TOKEN_BUCKET, PATCH_ADD_REGISTER)
    lines = [l.strip() for l in out.splitlines()]
    # declaration lands right after the port list closes
    assert lines[lines.index(");") + 1] == "reg [3:0] spent_total;"
    # reset lands inside the if(rst) block, after tokens_hi reset
    assert lines[lines.index("tokens_hi <= 1'b0;") + 1] == "spent_total <= 4'd0;"


def test_modify_always_patch_replaces_line():
    out = apply_patch(TOKEN_BUCKET, PATCH_MODIFY_ALWAYS)
    assert "tokens >= 3'd1" in out
    assert "tokens == 3'd1" not in out
    # one line swapped for one line
    assert len(out.splitlines()) == len(TOKEN_BUCKET.splitlines())


def test_malformed_patch_fails_cleanly():
    with pytest.raises(PatchError) as exc:
        apply_patch(TOKEN_BUCKET, PATCH_MALFORMED)
    # error names the anchor that failed to match
    assert "this line exists nowhere" in str(exc.value)


def test_failure_never_yields_half_a_file():
    # valid delete, then an impossible anchor: the delete must not leak
    diff = """\
- wire dec = spend_en && !drip_en;
@@ no such anchor line @@
+     reg ghost;"""
    with pytest.raises(PatchError):
        apply_patch(TOKEN_BUCKET, diff)
    # apply_patch raised, so the caller still holds the original


# --- strictness: every silent-skip in the original is an error here ---

def test_unmatched_delete_raises():
    with pytest.raises(PatchError, match="delete"):
        apply_patch("a\nb\n", "- not present")


def test_unmatched_keep_raises():
    with pytest.raises(PatchError, match="keep"):
        apply_patch("a\nb\n", "= not present")


def test_unknown_directive_raises():
    with pytest.raises(PatchError, match="directive"):
        apply_patch("a\nb\n", "* what is this")


def test_empty_patch_raises():
    with pytest.raises(PatchError, match="empty"):
        apply_patch("a\nb\n", "\n\n")


def test_anchors_only_patch_raises():
    with pytest.raises(PatchError, match="no operations"):
        apply_patch("a\nb\n", "@@ a @@")


# --- the yosys gate: patched designs must still read ---

@pytest.mark.skipif(shutil.which("yosys") is None,
                    reason="yosys not on PATH (run hwtools)")
@pytest.mark.parametrize("patch", [PATCH_ADD_REGISTER, PATCH_MODIFY_ALWAYS],
                         ids=["add-register", "modify-always"])
def test_patched_design_reads_in_yosys(patch):
    out = apply_patch(TOKEN_BUCKET, patch)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "design.sv"
        f.write_text(out)
        r = subprocess.run(
            ["yosys", "-p",
             f"read_verilog -sv {f}; hierarchy -top token_bucket; proc"],
            capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
