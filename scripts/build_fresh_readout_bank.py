"""Compile unmeasured fresh inputs, exact splits and an opaque agent-review packet."""

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from latent_probes.fresh_readout_bank import (
    ROOT,
    _fresh_row,
    audit_rows,
    build_bank,
    canonical,
    digest,
    iter_rows,
    review_packet,
    write_new,
)
from latent_probes.main_study_design import BeliefCase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dev",
        type=Path,
        default=ROOT / "data/belief_main/fresh_readout_v2/authored_train_dev_contexts_v1.json",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "data/belief_main/fresh_readout_v2/authored_test_contexts_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = (
        "bank.json",
        "split_manifest.json",
        "review_packet.json",
        "review_key.json",
        "engineering_row.json",
        "build_receipt.json",
    )
    assert not any((args.output / name).exists() for name in names), "Use fresh output filenames"
    bank = build_bank(args.train_dev, args.test)
    checked = audit_rows(bank)
    rows = list(iter_rows(bank))
    write_new(args.output / "bank.json", bank)
    packet, key = review_packet(bank)
    packet.update(bank_sha256=canonical(bank), bank_file_sha256=digest(args.output / "bank.json"))
    manifest = {
        "schema": "fresh-readout-split-manifest-v2",
        "bank_sha256": canonical(bank),
        "human_review_complete": False,
        "model_observations": 0,
        "partitions": {
            part: [r["record_id"] for r in rows if r["partition"] == part]
            for part in checked["partitions"]
        },
        "groups": [{"group_id": g["group_id"], "partition": g["split"]} for g in bank["groups"]],
        "ordered_prompt_identity_sha256": canonical(
            [{k: r[k] for k in ("record_id", "messages", "user_content_char_spans")} for r in rows]
        ),
        "fresh_test_strata": {
            str(k): v
            for k, v in Counter(
                (r["condition"], r["renderer"]) for r in rows if r["partition"] == "test"
            ).items()
        },
    }
    group = {
        "group_id": "fresh_engineering_excluded_v2",
        "split": "engineering",
        "sender": "Wesley Maddison",
        "reader": "Susannah Fenwick",
        "locations": ["striped cable tray", "plain routing drawer"],
        "introduction": "In a fictional routing-paperwork rehearsal, {sender} and {reader} track a token. This example is excluded from every scientific cohort.",
        "event_description": "The rehearsal provides a private written location channel to {sender}.",
    }
    engineering = _fresh_row(
        group,
        asdict(BeliefCase(group["group_id"], "correction", 0, 1, 0, 1, 0, 1)),
        "A",
        "correction_hard",
    )
    for name, value in (
        ("split_manifest.json", manifest),
        ("review_packet.json", packet),
        ("review_key.json", key),
        ("engineering_row.json", engineering),
    ):
        write_new(args.output / name, value)
    write_new(
        args.output / "build_receipt.json",
        {
            "schema": "fresh-readout-build-receipt-v2",
            "audit": checked,
            "review_prompts": len(packet["prompts"]),
            "review_base_contexts": len(packet["base_contexts"]),
            "output_sha256": {
                name: digest(args.output / name) for name in names if name != "build_receipt.json"
            },
        },
    )
    print(checked)


if __name__ == "__main__":
    main()
