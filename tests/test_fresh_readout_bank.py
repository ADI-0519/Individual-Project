"""Independent truth-table, split and input-identity checks for the fresh bank."""

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product

import pytest

from latent_probes import fresh_readout_bank as bankmod
from latent_probes.main_study_bank import iter_rows as old_rows
from latent_probes.main_study_bank import load_bank as old_bank

PATH = bankmod.ROOT / "data/belief_main/fresh_readout_v2/compiled_v1/bank.json"


@pytest.fixture(scope="module")
def bank():
    return bankmod.load_bank(PATH)


@pytest.fixture(scope="module")
def rows(bank):
    return list(bankmod.iter_rows(bank))


def test_inventory_groups_and_actual_label_rules(bank, rows):
    assert bankmod.audit_rows(bank)["rows"] == 7040
    assert Counter(r["partition"] for r in rows) == Counter(bankmod.COUNTS)
    assert len({r["record_id"] for r in rows}) == 7040
    for row in rows:
        state = row["state"]
        if state["mechanism"] == "asserted":
            expected = state["asserted"]
        elif state["mechanism"] == "correction":
            expected = (
                1 - state["update"]
                if state["receipt"]
                else state["update"]
                if state["first_receipt"]
                else state["initial"]
            )
        else:
            expected = state["update"] if state["receipt"] else state["initial"]
        assert row["gold"] == expected


def test_source_messages_ids_and_states_are_exact_originals(bank, rows):
    original = {
        r["case_id"]: r for r in old_rows(old_bank(bankmod.ROOT / bank["source_bank"]["path"]))
    }
    for row in rows:
        if row["partition"].startswith("source_"):
            before = original[row["case_id"]]
            for key in ("messages", "state", "gold", "group_id", "mechanism"):
                assert row[key] == before[key]
            assert row["original_split"] == before["split"]
    assert [r["case_id"] for r in rows if r["partition"] == "source_fit"] == bank[
        "source_fit_case_ids"
    ]


def test_whole_fresh_contexts_never_cross_cohorts_and_renderers_pair(rows):
    cohorts, paired = defaultdict(set), defaultdict(dict)
    for row in rows:
        cohorts[row["group_id"]].add(row["partition"])
        if row["partition"] == "test":
            identity = (row["group_id"], row["condition"], bankmod.canonical(row["state"]))
            paired[identity][row["renderer"]] = row
    assert all(len(values) == 1 for values in cohorts.values())
    assert len(paired) == 2496
    for values in paired.values():
        assert set(values) == {"A", "B"}
        assert values["A"]["gold"] == values["B"]["gold"]
        assert values["A"]["messages"] != values["B"]["messages"]


def test_parity_masks_retain_complete_receipt_panels_and_balanced_paths(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["partition"] == "correction_fit":
            grouped[row["group_id"]].append(row)
    for group_id, values in grouped.items():
        assert len(values) == 32
        paths = Counter(
            tuple(r["state"][k] for k in ("initial", "update", "first_receipt", "receipt"))
            for r in values
        )
        assert set(paths) == set(product((0, 1), repeat=4))
        assert set(paths.values()) == {2}
        panels = defaultdict(set)
        for row in values:
            s = row["state"]
            assert (
                s["initial"] ^ s["update"] ^ s["world"] ^ s["name_map"]
                == int(group_id.rsplit("_", 1)[1]) % 2
            )
            panels[(s["initial"], s["update"], s["world"], s["name_map"])].add(
                (s["first_receipt"], s["receipt"])
            )
        assert len(panels) == 8
        assert all(v == set(product((0, 1), repeat=2)) for v in panels.values())


def test_exact_spans_select_final_question_location_not_candidate(bank, rows):
    groups = {g["group_id"]: g for g in bank["groups"]}
    for row in rows:
        content = row["messages"][1]["content"]
        account, query = (row["user_content_char_spans"][k] for k in ("account", "query_location"))
        assert content[account[1] - 2 : account[1]] == "\n\n"
        assert "?" not in content[query[0] : query[1]]
        assert content[query[1] :].startswith("?")
        assert "YES" in content[query[1] :] and "NO" in content[query[1] :]
        if row["partition"] == "test":
            expected = bankmod.locations(groups[row["group_id"]], row["state"]["name_map"])[1]
            assert content[query[0] : query[1]] == expected
        assert "proposition" in content[: account[1]].lower()


def test_actor_pools_are_balanced_without_context_or_entity_reuse(bank):
    by_group = {g["group_id"]: g["split"] for g in bank["groups"]}
    origins = Counter(
        (by_group[r["group_id"]], r["author_pool"]) for r in bank["identity_assignment"]["records"]
    )
    assert origins == Counter(
        {
            ("correction_fit", 0): 8,
            ("correction_fit", 1): 8,
            ("correction_dev", 0): 4,
            ("correction_dev", 1): 4,
            ("test", 0): 12,
            ("test", 1): 12,
        }
    )
    assert len({r["actor_pair_origin"] for r in bank["identity_assignment"]["records"]}) == 48


def test_review_packet_hides_encoded_ids_state_and_gold(bank, rows):
    packet, key = bankmod.review_packet(bank)
    assert len(packet["base_contexts"]) == 48 and len(packet["prompts"]) == 108
    actual = {r["record_id"]: r for r in rows}
    lookup = {r["review_id"]: r for r in key["records"]}
    for prompt in packet["prompts"]:
        assert set(prompt) == {"review_id", "messages"}
        assert prompt["review_id"].startswith("R")
        original = actual[lookup[prompt["review_id"]]["record_id"]]
        assert prompt["messages"] == original["messages"]
        assert lookup[prompt["review_id"]]["gold"] == original["gold"]


@pytest.mark.parametrize("change", ["source_ids", "actor", "context", "c2"])
def test_fails_closed_on_binding_or_policy_changes(bank, change):
    altered = deepcopy(bank)
    if change == "source_ids":
        altered["source_fit_case_ids"][0] = altered["source_dev_case_ids"][0]
    elif change == "actor":
        altered["groups"][0]["sender"] = altered["groups"][1]["sender"]
    elif change == "context":
        altered["groups"][0]["introduction"] += " Unrecorded extra evidence."
    else:
        altered["C2"] = "rbf_svc"
    with pytest.raises(ValueError):
        bankmod.validate_bank(altered)
