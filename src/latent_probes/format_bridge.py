"""Cross fixed system/account/question packages without collecting model outputs.

These inputs reuse exposed contexts. New hybrid prompts are posthoc diagnostic
interventions, never a fresh confirmation set or independent reasoning families.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json

CONDITIONS = ("correction_easy", "correction_hard")
VARIANTS = tuple("".join(bits) for bits in itertools.product("AB", repeat=3))


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def identity(row):
    """Full formal state, not ordering, a shortened ID, or only the target label."""
    return row["group_id"], row["condition"], canonical(row["state"])


def _components(row):
    messages = row["messages"]
    require(
        len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"],
        "Require one system and one user donor message",
    )
    require(all(isinstance(m["content"], str) and m["content"] for m in messages), "Empty donor")
    content = messages[1]["content"]
    spans = row["user_content_char_spans"]
    require(set(spans) == {"account", "query_location"}, "Unexpected donor spans")
    account, query = spans["account"], spans["query_location"]
    require(
        all(
            isinstance(span, (list, tuple)) and len(span) == 2 and all(type(x) is int for x in span)
            for span in (account, query)
        ),
        "Require integer half-open spans",
    )
    require(
        0 == account[0] < account[1] <= query[0] < query[1] <= len(content),
        "Overlapping or out-of-range donor spans",
    )
    split = account[1]
    require(split < len(content), "Missing question")
    return {
        "system": messages[0]["content"],
        "account": content[:split],
        "question": content[split:],
        "query_span": [query[0] - split, query[1] - split],
        "query_location": content[query[0] : query[1]],
    }


def paired_donors(rows):
    """Require both original formats for every eligible exposed correction row."""
    pairs = {}
    for row in rows:
        if row.get("partition") != "test" or row.get("condition") not in CONDITIONS:
            continue
        require(row.get("renderer") in ("A", "B"), "Unexpected source renderer")
        key = identity(row)
        pair = pairs.setdefault(key, {})
        require(row["renderer"] not in pair, "Duplicate full-state donor")
        pair[row["renderer"]] = row
    require(bool(pairs), "No correction donors")
    for pair in pairs.values():
        require(set(pair) == {"A", "B"}, "Missing paired donor")
        a, b = pair["A"], pair["B"]
        require(type(a["gold"]) is int and a["gold"] in (0, 1), "Require integer binary gold")
        require(type(b["gold"]) is int and b["gold"] in (0, 1), "Require integer binary gold")
        require(a["gold"] == b["gold"], "Paired gold mismatch")
        require(a["mechanism"] == b["mechanism"] == "correction", "Wrong mechanism")
        state = a["state"]
        require(state["group_id"] == a["group_id"], "State/group identity mismatch")
        require(state["mechanism"] == "correction", "Wrong state mechanism")
        require(
            all(
                type(state[k]) is int and state[k] in (0, 1)
                for k in ("initial", "update", "first_receipt", "receipt", "world", "name_map")
            ),
            "Require binary correction state",
        )
        expected = (
            1 - state["update"]
            if state["receipt"]
            else state["update"]
            if state["first_receipt"]
            else state["initial"]
        )
        require(a["gold"] == expected, "Donor gold disagrees with formal update rule")
        require(
            (state["initial"] != state["update"]) == (a["condition"] == "correction_hard"),
            "Easy/hard condition disagrees with state",
        )
        components = {key: _components(value) for key, value in pair.items()}
        require(
            components["A"]["query_location"] == components["B"]["query_location"],
            "Paired queried physical location differs",
        )
        require(
            a.get("span_boundary_policy") == b.get("span_boundary_policy"), "Span policy differs"
        )
    return [pairs[key] for key in sorted(pairs)]


def cross_pair(pair):
    """Variant letters select system, complete account, and question respectively."""
    a, b = pair["A"], pair["B"]
    require(identity(a) == identity(b) and a["gold"] == b["gold"], "Unmatched donor pair")
    parts = {key: _components(row) for key, row in pair.items()}
    require(parts["A"]["query_location"] == parts["B"]["query_location"], "Query mismatch")
    result = []
    state_hash = hashlib.sha256(canonical(a["state"]).encode("utf-8")).hexdigest()[:16]
    for variant in VARIANTS:
        system, narrative, question = (parts[key] for key in variant)
        content = narrative["account"] + question["question"]
        split = len(narrative["account"])
        spans = {
            "account": [0, split],
            "query_location": [split + p for p in question["query_span"]],
        }
        record_id = f"{a['group_id']}__bridge_{a['condition']}_{variant}__{state_hash}"
        # No token IDs, scores, or hashes from a prepared/captured donor survive.
        row = {
            **{
                key: copy.deepcopy(a[key])
                for key in (
                    "group_id",
                    "partition",
                    "split",
                    "condition",
                    "mechanism",
                    "state",
                    "gold",
                )
            },
            "record_id": record_id,
            "case_id": record_id,
            "row_id": record_id,
            "renderer": variant,
            "format_components": dict(
                zip(("system", "narrative", "question"), variant, strict=True)
            ),
            "source_case_ids": {key: donor["case_id"] for key, donor in pair.items()},
            "messages": [
                {"role": "system", "content": system["system"]},
                {"role": "user", "content": content},
            ],
            "user_content_char_spans": spans,
            "span_boundary_policy": copy.deepcopy(a.get("span_boundary_policy", {})),
            "research_status": "posthoc_exposed_context_diagnostic_inputs",
            "human_review_complete": False,
        }
        if variant in ("AAA", "BBB"):
            original = pair[variant[0]]
            require(row["messages"] == original["messages"], "Diagonal changed original prompt")
            require(spans == original["user_content_char_spans"], "Diagonal changed original spans")
        result.append(row)
    return result


def build_rows(source_rows):
    rows = [row for pair in paired_donors(source_rows) for row in cross_pair(pair)]
    require(len({r["record_id"] for r in rows}) == len(rows), "Bridge row identity collision")
    return rows


def check_query_prefix_invariance(prepared_rows):
    """Fail if Q alone changes any token through account_last in a causal prefix.

    Tokenization at a changed suffix boundary can invalidate the presumed causal
    control. This checks actual complete-prefix tokens, not just account text.
    """
    indexed = {}
    for row in prepared_rows:
        key = (*identity(row), row["renderer"][:2])
        variants = indexed.setdefault(key, {})
        q = row["renderer"][2]
        require(q in "AB" and q not in variants, "Duplicate query-prefix control")
        position = row["token_positions"]["account_last"]
        variants[q] = (position, row["prompt_token_ids"][: position + 1])
    require(bool(indexed), "No query-prefix controls")
    for values in indexed.values():
        require(set(values) == {"A", "B"}, "Missing query-prefix counterpart")
        require(values["A"] == values["B"], "Query change alters causal account prefix")
    return len(indexed)
