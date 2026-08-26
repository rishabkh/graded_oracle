"""Strict applier for Formal Disco's line-based diff format.

Format (one directive per line):
  @@ text @@   anchor — consecutive anchors form a sequence; search forward
               from the cursor for consecutive lines matching them all, then
               set the cursor just past the last one. @@@@ is a no-op.
  = text       keep — search forward for the line, move cursor past it
  - text       delete — search forward for the line, delete it
  + text       add — insert at the cursor, advance past the insertion

Matching compares stripped text, exactly like formal-disco's
apply_text_diff. The one deliberate difference: THERE ARE NO SILENT
SKIPS. Their applier ignores a directive whose line is not found, which
is how a drifted patch half-applies into garbage (the Stage-2 Fixer
failure mode). Here every unmatched directive raises PatchError with
the offending line named, and the caller keeps the original text —
a malformed patch can never yield half a file.

Patches, not rewrites: lineage (parent vs child is always a readable
diff) and containment (a rewrite lets the model quietly regenerate the
whole module, which erases the difference between extending and
initiating).
"""


class PatchError(Exception):
    """A directive failed to match, or the patch itself is malformed."""


def apply_patch(text, diff):
    """Apply `diff` to `text`; return the patched text or raise PatchError.

    Never returns a partial result: any failure raises, and `text` is
    untouched (all edits happen on a private copy).
    """
    lines = text.splitlines(keepends=True)
    cursor = 0

    def content(i):
        return lines[i].rstrip("\n")

    def find_forward(target, start):
        for idx in range(start, len(lines)):
            if content(idx).strip() == target.strip():
                return idx
        return None

    def find_sequence(anchors, start):
        for idx in range(start, len(lines) - len(anchors) + 1):
            if all(content(idx + j).strip() == anchors[j].strip()
                   for j in range(len(anchors))):
                return idx
        return None

    diff_lines = diff.splitlines()
    if not any(line.strip() for line in diff_lines):
        raise PatchError("empty patch: no directives")

    pending_anchors = []
    n_ops = 0

    def flush_anchors():
        nonlocal cursor, pending_anchors
        if not pending_anchors:
            return
        j = find_sequence(pending_anchors, cursor)
        if j is None:
            raise PatchError(
                f"anchor sequence not found after line {cursor}: "
                + " / ".join(repr(a.strip()) for a in pending_anchors))
        cursor = j + len(pending_anchors)
        pending_anchors = []

    for raw in diff_lines:
        if not raw.strip():
            continue
        if raw.startswith("@@") and raw.endswith("@@") and len(raw) >= 4:
            anchor = raw[2:-2]
            if anchor.strip():
                pending_anchors.append(anchor)
            continue   # @@@@ (or all-whitespace anchor) is a no-op sync point

        flush_anchors()

        op, payload = raw[0], raw[1:]
        if payload.startswith(" "):
            payload = payload[1:]

        if op == "=":
            j = find_forward(payload, cursor)
            if j is None:
                raise PatchError(f"keep line not found: {payload.strip()!r}")
            cursor = j + 1
        elif op == "-":
            j = find_forward(payload, cursor)
            if j is None and cursor > 0 and \
                    content(cursor - 1).strip() == payload.strip():
                # Anchor-then-delete of the same line: the anchor already
                # consumed it, so forward search misses. The intent is
                # unambiguous — delete the just-anchored line.
                j = cursor - 1
            if j is None:
                raise PatchError(f"delete line not found: {payload.strip()!r}")
            del lines[j]
            cursor = j
        elif op == "+":
            lines.insert(cursor, payload + "\n")
            cursor += 1
        else:
            raise PatchError(f"unknown directive: {raw!r}")
        n_ops += 1

    flush_anchors()   # trailing anchors must still match — strictness applies
    if n_ops == 0:
        raise PatchError("patch contains no operations (+/-/=)")
    return "".join(lines)
