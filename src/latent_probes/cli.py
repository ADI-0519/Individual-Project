"""Command line entry points. Planning and data checks do not require a GPU."""

import argparse
import json
import sys

from .backends import doctor
from .config import load_config
from .experiment import export_annotations, plan, replay_recipients, run_senders
from .features import pack_features
from .import_examples import extract_examples
from .reporting import summarise_recipients
from .schema import load_scenarios


def parser():
    p = argparse.ArgumentParser(
        description="Latent-probe research starter. See README.md for the staged workflow."
    )
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Inspect environment and run a tiny CUDA kernel if available"
    )
    validate = commands.add_parser(
        "validate-data", help="Check draft scenario structure and balance"
    )
    validate.add_argument("--scenarios", default="data/pilot_scenarios.jsonl")
    for name in ("plan", "run"):
        cmd = commands.add_parser(
            name,
            help="Export prompts without model execution"
            if name == "plan"
            else "Generate sender messages and activations",
        )
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--scenarios", default="data/pilot_scenarios.jsonl")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--limit", type=int)
        if name == "run":
            cmd.add_argument("--resume", action="store_true")
    replay = commands.add_parser(
        "replay-recipients", help="Replay saved messages to independent recipient contexts"
    )
    replay.add_argument("--run", required=True)
    replay.add_argument(
        "--config", help="Override recipient model/settings while keeping sender messages fixed"
    )
    replay.add_argument("--output")
    replay.add_argument("--resume", action="store_true")
    summarise = commands.add_parser(
        "summarise", aliases=["summarize"], help="Summarise recipient outcomes and controls"
    )
    summarise.add_argument("--recipients", required=True)
    summarise.add_argument("--output", required=True)
    annotate = commands.add_parser(
        "export-annotations", help="Export blank manual annotation fields"
    )
    annotate.add_argument("--run", required=True)
    annotate.add_argument("--output", required=True)
    pack = commands.add_parser(
        "pack-features", help="Select one layer and token position for a probe"
    )
    pack.add_argument("--run", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--layer", required=True, type=int)
    pack.add_argument(
        "--position", choices=["prompt_last", "response_last"], default="response_last"
    )
    pack.add_argument("--annotations")
    extract = commands.add_parser(
        "extract-examples", help="Generate/capture independent probe-training tasks"
    )
    extract.add_argument("--config", required=True)
    extract.add_argument("--examples", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--resume", action="store_true")
    train = commands.add_parser(
        "train-probe", help="Fit a regularised probe with grouped held-out evaluation"
    )
    train.add_argument("--features", required=True)
    train.add_argument("--labels", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--target", default="label")
    train.add_argument("--group-column", default="family_id")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--bootstrap-samples", type=int, default=300)
    predict = commands.add_parser(
        "predict-probe", help="Apply a saved probe with capture-signature checking"
    )
    predict.add_argument("--features", required=True)
    predict.add_argument("--probe", required=True)
    predict.add_argument("--output", required=True)
    apollo = commands.add_parser("apollo", help="Source-audited token-level Apollo adaptation")
    stages = apollo.add_subparsers(dest="apollo_command", required=True)
    fetch = stages.add_parser("fetch-facts", help="Download the pinned upstream CSV explicitly")
    fetch.add_argument("--output", required=True)
    prepare = stages.add_parser("prepare", help="Prepare ordered, pair-disjoint instruction data")
    prepare.add_argument("--facts", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int, default=512)
    prepare.add_argument("--short-fact-policy", choices=["error", "exclude"], default="error")
    prepare.add_argument("--duplicate-fact-policy", choices=["error", "exclude"], default="error")
    extract_tokens = stages.add_parser(
        "extract", help="Capture masked token rows on a real checkpoint"
    )
    extract_tokens.add_argument("--config", required=True)
    extract_tokens.add_argument("--examples", required=True)
    extract_tokens.add_argument("--output", required=True)
    extract_tokens.add_argument("--resume", action="store_true")
    for stage in ("train", "score"):
        cmd = stages.add_parser(stage)
        cmd.add_argument("--features", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--labels" if stage == "train" else "--probe", required=True)
    calibrate = stages.add_parser(
        "calibrate", help="Independent neutral-control raw-score percentile"
    )
    calibrate.add_argument("--scores", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--quantile", type=float, default=0.99)
    evaluate = stages.add_parser("evaluate", help="Evaluate explicitly audited response labels")
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--threshold")
    return p


def main(argv=None):
    args = parser().parse_args(argv)

    def progress(message):
        print(message, file=sys.stderr, flush=True)

    try:
        if args.command == "apollo":
            from .apollo import extract_token_examples, fetch_facts, prepare_pairs
            from .apollo_probe import (
                calibrate_control_threshold,
                evaluate_apollo_scores,
                score_apollo_probe,
                train_apollo_probe,
            )

            stage = args.apollo_command
            if stage == "fetch-facts":
                result = fetch_facts(args.output)
            elif stage == "prepare":
                result = prepare_pairs(
                    args.facts,
                    args.output,
                    args.limit,
                    short_fact_policy=args.short_fact_policy,
                    duplicate_fact_policy=args.duplicate_fact_policy,
                )
            elif stage == "extract":
                result = extract_token_examples(
                    load_config(args.config),
                    args.examples,
                    args.output,
                    args.resume,
                    progress=progress,
                )
            elif stage == "train":
                result = train_apollo_probe(args.features, args.labels, args.output)
            elif stage == "score":
                result = score_apollo_probe(args.features, args.probe, args.output)
            elif stage == "calibrate":
                result = calibrate_control_threshold(args.scores, args.output, args.quantile)
            else:
                result = evaluate_apollo_scores(
                    args.scores, args.labels, args.output, args.threshold
                )
        elif args.command == "doctor":
            result = doctor()
        elif args.command == "validate-data":
            data = load_scenarios(args.scenarios)
            result = {
                "scenarios": len(data),
                "families": len({s.family_id for s in data}),
                "domains": sorted({s.domain for s in data}),
                "full_information_answers": {
                    a: sum(s.full_information_answer == a for s in data) for a in ("yes", "no")
                },
                "status": "structure_valid; semantic_human_review_still_required",
            }
        elif args.command == "plan":
            result = plan(load_config(args.config), args.scenarios, args.output, args.limit)
        elif args.command == "run":
            result = run_senders(
                load_config(args.config),
                args.scenarios,
                args.output,
                args.limit,
                args.resume,
                progress=progress,
            )
        elif args.command == "replay-recipients":
            config = load_config(args.config) if args.config else None
            result = replay_recipients(
                args.run, config, args.output, args.resume, progress=progress
            )
        elif args.command in {"summarise", "summarize"}:
            result = summarise_recipients(args.recipients, args.output)
        elif args.command == "export-annotations":
            result = export_annotations(args.run, args.output)
        elif args.command == "pack-features":
            result = pack_features(
                args.run, args.output, args.layer, args.position, args.annotations
            )
        elif args.command == "extract-examples":
            result = extract_examples(
                load_config(args.config),
                args.examples,
                args.output,
                args.limit,
                args.resume,
                progress=progress,
            )
        elif args.command == "train-probe":
            from .probes import train_probe

            result = train_probe(
                args.features,
                args.labels,
                args.output,
                target=args.target,
                group_column=args.group_column,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
        elif args.command == "predict-probe":
            from .probes import predict_probe

            result = predict_probe(args.features, args.probe, args.output)
        else:
            raise ValueError("Unknown command")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ImportError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
