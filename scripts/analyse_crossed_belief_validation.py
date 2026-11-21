"""Verify frozen crossed-belief raw evidence and rebuild its descriptive record."""

import argparse
from pathlib import Path

from latent_probes.crossed_belief_analysis import analyse
from latent_probes.storage import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--verify-tokenizer", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Analysis output must stay outside the byte-bound raw run")
    tokenizer = None
    if args.verify_tokenizer:
        from transformers import AutoTokenizer

        settings = read_json(args.run / "protocol.json")["model_settings"]
        tokenizer = AutoTokenizer.from_pretrained(
            settings["model_id"],
            revision=settings["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    record = analyse(args.run, tokenizer=tokenizer, native=args.native)
    atomic_json(args.output, record)
    query = record["conditions"]["nested_question"]
    print(
        {
            "status": record["status"],
            "primary": query["primary"],
            "probe_qualification": query["probe_qualification"],
            "output_ranking_qualification": query["output_ranking_qualification"],
            "behavioral_qualification": query["behavioral_qualification"],
        }
    )


if __name__ == "__main__":
    main()
