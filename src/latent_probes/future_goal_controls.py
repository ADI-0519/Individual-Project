"""Material checks and the fixed neutral feasibility gate; no model fitting."""

from __future__ import annotations

from collections import Counter

from .storage import file_sha256, read_json

BANK_SHA256 = "8c1b64638d12a93585e1bef8260388437ca2404feff6809a214f18a598e78a79"
HEAD = "nested_question__sender__nested__layer_16"
CONDITIONS = (
    "neutral",
    "own_token_a",
    "own_token_b",
    "own_disk_a",
    "own_disk_b",
    "archive_token_a",
    "archive_token_b",
)
GATE = {"overall": [24, 32], "per_mapping": [12, 16], "per_channel": [10, 16]}


def require(value, message):
    if not value:
        raise ValueError(message)


def load_material(path):
    """Accept only the reviewed bytes, and rederive the authored source labels."""
    require(file_sha256(path) == BANK_SHA256, "Future-goal bank SHA256 differs")
    bank = read_json(path)
    require(bank["schema"] == "future_goal_content_controls_draft_v1", "Wrong bank schema")
    require(bank["primary_head"] == HEAD, "Frozen head differs")
    require(tuple(bank["conditions"]) == CONDITIONS, "Condition roster differs")
    scenes, rows = bank["scenes"], bank["planned_conditions"]
    require(len(scenes) == 16 and len(rows) == 224, "Need 16 scenes and 224 conditions")
    by_scene = {scene["id"]: scene for scene in scenes}
    require(len(by_scene) == 16, "Duplicate scene IDs")
    require(
        Counter(s["channel"] for s in scenes) == {"observation": 8, "message": 8},
        "Evidence-channel roster differs",
    )
    groups = {}
    for scene in scenes:
        source = scene["source"]
        update = (
            source["world_slot"]
            if scene["channel"] == "observation"
            else source["update_claim_slot"]
        )
        attributed = update if source["receipt_reports_update"] else source["initial_slot"]
        require(
            scene["claim_slot"] == 1
            and scene["current_attributed_slot"] == attributed
            and scene["gold"] == int(attributed == 1),
            "Source-rule label differs",
        )
        groups.setdefault(scene["wording_group"], []).append(scene["gold"])
    require(
        len(groups) == 8 and all(sorted(v) == [0, 1] for v in groups.values()),
        "Need eight opposing-label wording pairs",
    )
    indexed = {}
    for row in rows:
        scene = by_scene[row["scene_id"]]
        key = (row["scene_id"], row["mapping_id"], row["condition"])
        require(
            key not in indexed and row["mapping_id"] in (0, 1) and row["condition"] in CONDITIONS,
            "Duplicate or unexpected condition",
        )
        require(row["id"] == f"{key[0]}_map{key[1]}_{key[2]}", "Row ID differs")
        require(
            all(row[k] == scene[k] for k in ("channel", "wording_group", "gold", "claim_slot")),
            "Scene/row metadata differs",
        )
        places = scene["physical_names"][:: (-1 if row["mapping_id"] else 1)]
        require([row["place_a"], row["place_b"]] == places, "Physical mapping differs")
        require(
            [m["role"] for m in row["messages"]] == ["system", "user"],
            "Only exact system/user messages are permitted",
        )
        indexed[key] = row
    for scene in scenes:
        for mapping in (0, 1):
            selected = [indexed[(scene["id"], mapping, c)] for c in CONDITIONS]
            require(
                len({r["messages"][1]["content"] for r in selected}) == 1,
                "User scene changes across contexts",
            )
            for slot in ("a", "b"):
                own = indexed[(scene["id"], mapping, "own_token_" + slot)]
                archive = indexed[(scene["id"], mapping, "archive_token_" + slot)]
                require(
                    own["inner_objective"] == archive["inner_objective"],
                    "Own/archive inner objective differs",
                )
        a, b = scene["physical_names"]
        for condition in CONDITIONS:
            left, right = (indexed[(scene["id"], m, condition)] for m in (0, 1))
            for x, y in zip(left["messages"], right["messages"], strict=True):
                exchanged = x["content"].replace(a, "\x00").replace(b, a).replace("\x00", b)
                require(exchanged == y["content"], "Mapping is not an exact global name exchange")
    require(
        Counter(r["condition"] for r in rows) == dict.fromkeys(CONDITIONS, 32),
        "Condition counts differ",
    )
    return bank


def qualify_neutral(rows):
    """Invalid/capped answers count as incorrect, with no rescue or retries."""
    require(
        len(rows) == 32 and all(r["condition"] == "neutral" for r in rows),
        "Qualification requires exactly 32 neutral outputs",
    )
    require(len({r["id"] for r in rows}) == 32, "Duplicate neutral outputs")
    failed = []

    def cell(selected, name, floor=None, expected=None):
        value = {
            "n": len(selected),
            "probe_correct": sum(r["probe"]["prediction"] == r["gold"] for r in selected),
            "answer_correct": sum(r["answer"]["prediction"] == r["gold"] for r in selected),
            "invalid": sum(r["answer"]["prediction"] is None for r in selected),
        }
        if floor is not None:
            require(value["n"] == expected, "Neutral subgroup roster differs: " + name)
            for metric in ("probe_correct", "answer_correct"):
                if value[metric] < floor:
                    failed.append(f"{name}/{metric}<{floor}/{expected}")
        return value

    result = {
        "schema": "future_goal_neutral_qualification_v1",
        "gate": GATE,
        "overall": cell(rows, "overall", *GATE["overall"]),
    }
    result["per_mapping"] = {
        str(m): cell(
            [r for r in rows if r["mapping_id"] == m], f"mapping_{m}", *GATE["per_mapping"]
        )
        for m in (0, 1)
    }
    result["per_channel"] = {
        c: cell([r for r in rows if r["channel"] == c], c, *GATE["per_channel"])
        for c in ("observation", "message")
    }
    result["crossed"] = {
        f"{c}/map{m}": cell(
            [r for r in rows if r["channel"] == c and r["mapping_id"] == m], f"{c}/map{m}"
        )
        for c in ("observation", "message")
        for m in (0, 1)
    }
    result.update(passed=not failed, failed_checks=failed)
    return result
