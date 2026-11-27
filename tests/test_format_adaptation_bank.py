"""Independent contract checks for the fresh, fixed-budget format allocation."""

import copy
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from latent_probes import format_adaptation_bank as design

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/belief_main/format_adaptation_v1"
PATH_FACTORS = ("initial", "update", "first_receipt", "receipt")
CORRECTION_FACTORS = (*PATH_FACTORS, "world", "name_map")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_key(row):
    return canonical(row["state"])


def independent_gold(row):
    """Follow accepted reports in event-time order; never call production label code."""
    state = row["state"]
    if row["condition"] == "asserted":
        attributed_location = state["asserted"]
    else:
        attributed_location = state["initial"]
        first_read = (
            state["first_receipt"] if row["mechanism"] == "correction" else state["receipt"]
        )
        reports = [(state["update"], first_read)]
        if row["mechanism"] == "correction":
            reports.append((1 - state["update"], state["receipt"]))
        for location, accepted_as_read in reports:
            if accepted_as_read:
                attributed_location = location
    return int(attributed_location == 1)


@pytest.fixture(scope="module")
def authored():
    return read(DATA / "authored_contexts_v1.json")


@pytest.fixture(scope="module")
def bank():
    return design.load(DATA / "compiled_v1/bank.json")


@pytest.fixture(scope="module")
def rows(bank):
    return list(design.iter_rows(bank))


def test_frozen_inputs_and_authoring_overlap_receipt_reproduce(authored, bank):
    receipt = read(DATA / "authoring_receipt_v1.json")
    assert (
        file_sha(ROOT / receipt["authored_context_file"]["path"])
        == (receipt["authored_context_file"]["sha256"])
    )
    prior = []
    for binding in receipt["comparison_inputs"]:
        path = ROOT / binding["path"]
        assert file_sha(path) == binding["sha256"]
        old = read(path)["groups"]
        assert len(old) == binding["groups"]
        prior.extend(old)
    extra = receipt["additional_engineering_entity_input"]
    assert file_sha(ROOT / extra["path"]) == extra["sha256"]
    assert all(name in (ROOT / extra["path"]).read_text() for name in extra["actor_names"])
    old_names = [g[role] for g in prior for role in ("sender", "reader")]
    old_names.extend(extra["actor_names"])
    old_components = {part.casefold() for name in old_names for part in name.split()}
    old_locations = {p.casefold() for g in prior for p in g["locations"]}
    new_names = [g[role] for g in authored["groups"] for role in ("sender", "reader")]
    new_components = {part.casefold() for name in new_names for part in name.split()}
    new_locations = {p.casefold() for g in authored["groups"] for p in g["locations"]}
    assert len(new_names) == len(set(new_names)) == 80
    assert len(new_components) == 160
    assert len(new_locations) == 80
    assert not {n.casefold() for n in new_names} & {n.casefold() for n in old_names}
    assert not new_components & old_components
    assert not new_locations & old_locations

    def pair_set(groups):
        return {tuple(sorted((g["sender"].casefold(), g["reader"].casefold()))) for g in groups}

    def framing_set(groups):
        return {
            " ".join((g["introduction"] + " " + g["event_description"]).casefold().split())
            for g in groups
        }

    recomputed = {
        "actor_full_name_overlap": sorted(
            {n.casefold() for n in new_names} & {n.casefold() for n in old_names}
        ),
        "actor_component_overlap": sorted(new_components & old_components),
        "unordered_actor_pair_overlap": [
            list(p) for p in sorted(pair_set(authored["groups"]) & pair_set(prior))
        ],
        "location_entity_overlap": sorted(new_locations & old_locations),
        "wording_group_overlap": sorted(
            {g["wording_group"] for g in authored["groups"]} & {g["wording_group"] for g in prior}
        ),
        "exact_normalized_full_framing_overlap": sorted(
            framing_set(authored["groups"]) & framing_set(prior)
        ),
    }
    assert all(not overlaps for overlaps in recomputed.values())
    assert receipt["overlap_results"] == recomputed
    assert receipt["prior_unique_entity_counts"] == {
        "actor_full_names": len({n.casefold() for n in old_names}),
        "actor_components": len(old_components),
        "physical_location_entities": len(old_locations),
        "actor_pairs": len(pair_set(prior)),
    }
    assert receipt["new_unique_entity_counts"] == {
        "actor_full_names": 80,
        "actor_components": 160,
        "physical_location_entities": 80,
        "actor_pairs": 40,
    }
    assert receipt["human_review_complete"] is False
    assert receipt["model_observations"] == 0
    assert authored["author_pool_id"] == receipt["author_pool_id"]
    assert bank["human_review_complete"] is False and bank["model_observations"] == 0
    for split in ("fit", "test"):
        cohort = [g for g in authored["groups"] if g["split"] == split]
        lengths = [len((g["introduction"] + " " + g["event_description"]).split()) for g in cohort]
        assert receipt["split_balance"][split] == {
            "groups": len(cohort),
            "domains": dict(Counter(g["domain_category"] for g in cohort)),
            "framing_templates": dict(Counter(g["framing_template_id"] for g in cohort)),
            "framing_word_count": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
        }


def test_compilation_preserves_contexts_and_permutates_only_actor_pairs(authored, bank):
    before = copy.deepcopy(authored)
    assert design.compile_bank(authored) == bank
    assert authored == before
    assert hashlib.sha256(canonical(authored).encode()).hexdigest() == bank["authoring_sha256"]
    source_by_id = {g["group_id"]: g for g in authored["groups"]}
    expected_pairs = sorted(
        [(g["sender"], g["reader"]) for g in authored["groups"]],
        key=lambda p: hashlib.sha256(
            canonical(["adaptation-actors-20261127", *p]).encode()
        ).hexdigest(),
    )
    assert [g["group_id"] for g in bank["groups"]] == sorted(source_by_id)
    assert [(g["sender"], g["reader"]) for g in bank["groups"]] == expected_pairs
    for group in bank["groups"]:
        source = source_by_id[group["group_id"]]
        assert {k: v for k, v in group.items() if k not in ("sender", "reader")} == {
            k: v for k, v in source.items() if k not in ("sender", "reader")
        }
    for split, expected_domain, expected_template in (("fit", 2, 4), ("test", 3, 6)):
        cohort = [g for g in bank["groups"] if g["split"] == split]
        assert len(Counter(g["domain_category"] for g in cohort)) == 8
        assert set(Counter(g["domain_category"] for g in cohort).values()) == {expected_domain}
        assert set(Counter(g["framing_template_id"] for g in cohort).values()) == {
            expected_template
        }


def test_fit_and_test_entities_are_disjoint_after_permutation(bank, rows):
    entities = {}
    for split in ("fit", "test"):
        cohort = [g for g in bank["groups"] if g["split"] == split]
        entities[split] = {
            value.casefold()
            for group in cohort
            for value in (group["sender"], group["reader"], *group["locations"])
        }
    assert entities["fit"].isdisjoint(entities["test"])
    assert {r["record_id"] for r in rows if r["partition"] == "fit"}.isdisjoint(
        r["record_id"] for r in rows if r["partition"] == "test"
    )


def test_exact_capture_and_condition_inventory(rows):
    assert len(rows) == 5760
    assert Counter(r["partition"] for r in rows) == {"fit": 768, "test": 4992}
    assert len({r["record_id"] for r in rows}) == len(rows)
    assert len({canonical(r["messages"]) for r in rows}) == len(rows)
    assert Counter((r["partition"], r["renderer"], r["condition"]) for r in rows) == {
        ("fit", "A", "correction_easy"): 256,
        ("fit", "A", "correction_hard"): 256,
        ("fit", "B", "correction_easy"): 128,
        ("fit", "B", "correction_hard"): 128,
        **{
            ("test", renderer, condition): count
            for renderer in ("A", "B")
            for condition, count in (
                ("correction_easy", 768),
                ("correction_hard", 768),
                ("one_step", 768),
                ("asserted", 192),
            )
        },
    }


def test_equal_budget_a_and_mixed_rosters_cover_identical_states(rows):
    fit = [r for r in rows if r["partition"] == "fit"]
    a_only = [r for r in fit if r["renderer"] == "A"]
    mixed = [r for r in fit if r["renderer"] == r["mixed_format"]]
    a_half = [r for r in mixed if r["renderer"] == "A"]
    b_half = [r for r in mixed if r["renderer"] == "B"]
    assert len(a_only) == len(mixed) == 512
    assert len(a_half) == len(b_half) == 256
    assert {state_key(r) for r in a_only} == {state_key(r) for r in mixed}
    assert len({state_key(r) for r in fit}) == 512
    for group_id in {r["group_id"] for r in fit}:
        raw = [r for r in fit if r["group_id"] == group_id]
        selected = [r for r in mixed if r["group_id"] == group_id]
        assert len(raw) == 48 and len(selected) == 32
        assert len({state_key(r) for r in raw}) == 32
        assert Counter(r["renderer"] for r in selected) == {"A": 16, "B": 16}
        for renderer in ("A", "B"):
            assert Counter(
                tuple(r["state"][k] for k in PATH_FACTORS)
                for r in selected
                if r["renderer"] == renderer
            ) == dict.fromkeys(itertools.product((0, 1), repeat=4), 1)
    cells = Counter((r["renderer"], *(r["state"][k] for k in CORRECTION_FACTORS)) for r in mixed)
    assert set(cells) == set(itertools.product(("A", "B"), *((0, 1),) * 6))
    assert len(cells) == 128 and set(cells.values()) == {4}


def test_every_test_context_has_complete_independent_state_inventory(rows):
    by_group = defaultdict(list)
    for row in rows:
        if row["partition"] == "test":
            by_group[row["group_id"]].append(row)
    assert len(by_group) == 24
    for values in by_group.values():
        assert len(values) == 208
        for renderer in ("A", "B"):
            subset = [r for r in values if r["renderer"] == renderer]
            correction = [r for r in subset if r["mechanism"] == "correction"]
            one_step = [r for r in subset if r["condition"] == "one_step"]
            asserted = [r for r in subset if r["condition"] == "asserted"]
            assert len(correction) == 64
            assert {tuple(r["state"][k] for k in CORRECTION_FACTORS) for r in correction} == (
                set(itertools.product((0, 1), repeat=6))
            )
            assert len(one_step) == 32
            assert {
                tuple(r["state"][k] for k in ("initial", "update", "receipt", "world", "name_map"))
                for r in one_step
            } == set(itertools.product((0, 1), repeat=5))
            assert len(asserted) == 8
            assert {
                tuple(r["state"][k] for k in ("asserted", "world", "name_map")) for r in asserted
            } == set(itertools.product((0, 1), repeat=3))


def test_all_gold_labels_follow_accepted_reports_and_actor_permutation_is_irrelevant(
    authored, bank, rows
):
    for row in rows:
        assert row["gold"] == independent_gold(row), row["record_id"]
    original_actors = {**bank, "groups": sorted(authored["groups"], key=lambda g: g["group_id"])}
    assert [(r["record_id"], r["gold"]) for r in design.iter_rows(original_actors)] == [
        (r["record_id"], r["gold"]) for r in rows
    ]


def test_correction_receipt_quartets_have_three_to_one_hard_labels(rows):
    quartets = defaultdict(list)
    for row in rows:
        if row["mechanism"] == "correction":
            state = row["state"]
            quartets[
                (
                    row["group_id"],
                    row["renderer"],
                    *(state[k] for k in ("initial", "update", "world", "name_map")),
                )
            ].append(row)
    assert quartets
    for members in quartets.values():
        assert len(members) == 4
        assert {(r["state"]["first_receipt"], r["state"]["receipt"]) for r in members} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        state = members[0]["state"]
        if state["initial"] != state["update"]:
            assert Counter(r["gold"] for r in members) == {state["initial"]: 3, state["update"]: 1}
            minority = [r for r in members if r["gold"] == state["update"]]
            assert [(r["state"]["first_receipt"], r["state"]["receipt"]) for r in minority] == [
                (1, 0)
            ]
        else:
            assert Counter(r["gold"] for r in members) == {0: 2, 1: 2}


def test_test_renderers_and_nuisance_variants_preserve_gold(rows):
    renderer_pairs = defaultdict(list)
    nuisance_groups = defaultdict(list)
    for row in rows:
        if row["partition"] != "test":
            continue
        renderer_pairs[state_key(row)].append(row)
        nuisance_groups[
            (
                row["renderer"],
                canonical(
                    {k: v for k, v in row["state"].items() if k not in ("world", "name_map")}
                ),
            )
        ].append(row)
    assert len(renderer_pairs) == 2496
    for pair in renderer_pairs.values():
        assert {r["renderer"] for r in pair} == {"A", "B"}
        assert len({r["gold"] for r in pair}) == 1
        assert pair[0]["messages"] != pair[1]["messages"]
    for group in nuisance_groups.values():
        assert len(group) == 4
        assert {(r["state"]["world"], r["state"]["name_map"]) for r in group} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        assert len({r["gold"] for r in group}) == 1


def test_prompt_spans_and_metadata_exposure(bank, rows):
    contexts = {g["group_id"]: g for g in bank["groups"]}
    for row in rows:
        group = contexts[row["group_id"]]
        assert [m["role"] for m in row["messages"]] == ["system", "user"]
        user = row["messages"][1]["content"]
        assert row["record_id"] not in user and row["group_id"] not in user
        assert not re.search(r"\b(?:gold|mixed_format|partition|record_id)\s*[:=]", user)
        assert group["sender"] in user and group["reader"] in user
        spans = row["user_content_char_spans"]
        account_start, account_end = spans["account"]
        query_start, query_end = spans["query_location"]
        expected_target = group["locations"][1 - row["state"]["name_map"]]
        assert account_start == 0 < account_end < query_start < query_end <= len(user)
        assert user[query_start:query_end] == expected_target
        account, separator, question = user.rpartition("\n\n")
        assert separator == "\n\n"
        assert account_end == len(account) + len(separator)
        assert expected_target in question
        assert row["span_boundary_policy"]["query_location"] == "leading_whitespace"
        assert row["gold"] in (0, 1)


def test_saved_build_receipt_binds_unchanged_bank_and_ordered_prompts(rows):
    compiled = DATA / "compiled_v1"
    receipt = read(compiled / "build_receipt.json")
    assert receipt["human_review_complete"] is False and receipt["model_observations"] == 0
    assert file_sha(DATA / "authored_contexts_v1.json") == receipt["authored_sha256"]
    # Never load the independent semantic review packet, its key or its answers.
    for filename in ("bank.json", "allocation_audit.json"):
        assert file_sha(compiled / filename) == receipt["output_sha256"][filename]
    prompt_identity = [
        {key: row[key] for key in ("record_id", "messages", "user_content_char_spans")}
        for row in rows
    ]
    assert (
        hashlib.sha256(canonical(prompt_identity).encode()).hexdigest()
        == (receipt["ordered_prompt_sha256"])
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unrelated-schema"),
        ("status", "measured"),
        ("human_review_complete", True),
        ("model_observations", 1),
    ],
)
def test_compile_rejects_invalid_or_measured_authored_metadata(authored, field, value):
    changed = copy.deepcopy(authored)
    changed[field] = value
    with pytest.raises(ValueError):
        design.compile_bank(changed)


@pytest.mark.parametrize("field,value", [("status", "measured"), ("model_observations", 1)])
def test_validate_rejects_invalid_or_measured_bank_metadata(bank, field, value):
    changed = copy.deepcopy(bank)
    changed[field] = value
    with pytest.raises(ValueError):
        design.validate(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "split_disagreement",
        "shared_name_part",
        "shared_place",
        "one_word_name",
        "non_ascii_name",
        "three_locations",
        "empty_introduction",
    ],
)
def test_structurally_invalid_contexts_are_rejected(bank, mutation):
    changed = copy.deepcopy(bank)
    first, second = changed["groups"][:2]
    if mutation == "duplicate_id":
        second["group_id"] = first["group_id"]
    elif mutation == "split_disagreement":
        first["split"], changed["groups"][-1]["split"] = "test", "fit"
    elif mutation == "shared_name_part":
        second["sender"] = first["sender"].split()[0] + " " + second["sender"].split()[1]
    elif mutation == "shared_place":
        second["locations"][0] = first["locations"][0]
    elif mutation == "one_word_name":
        first["sender"] = "Mononym"
    elif mutation == "non_ascii_name":
        first["sender"] = "Andr\u00e9 Novelname"
    elif mutation == "three_locations":
        first["locations"].append("spare reference chest")
    elif mutation == "empty_introduction":
        first["introduction"] = ""
    with pytest.raises(ValueError):
        design.validate(changed)


def test_unknown_partition_is_rejected(bank):
    with pytest.raises(ValueError):
        list(design.iter_rows(bank, "development"))
