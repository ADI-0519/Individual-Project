"""Freeze fresh authored inputs, exact allocation audit and opaque review packet."""

import argparse
import hashlib
import json
from pathlib import Path

from latent_probes import format_adaptation_bank as design
from latent_probes.storage import file_sha256, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    authored = json.loads(args.authored.read_text(encoding="utf-8"))
    bank = design.compile_bank(authored)
    audit = design.audit_rows(bank)
    packet, key = design.review_packet(bank)
    for name, value in (
        ("bank.json", bank),
        ("allocation_audit.json", audit),
        ("review_packet.json", packet),
        ("review_key.json", key),
    ):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    receipt = {
        "schema": "format-adaptation-input-build-v1",
        "created_at": utc_now(),
        "human_review_complete": False,
        "model_observations": 0,
        "authored_sha256": file_sha256(args.authored),
        "source_sha256": {
            str(Path(__file__).resolve()): file_sha256(__file__),
            str(Path(design.__file__).resolve()): file_sha256(design.__file__),
        },
        "ordered_prompt_sha256": hashlib.sha256(
            design.canonical(
                [
                    {k: r[k] for k in ("record_id", "messages", "user_content_char_spans")}
                    for r in design.iter_rows(bank)
                ]
            ).encode()
        ).hexdigest(),
        "review_items": len(packet["items"]),
        "output_sha256": {p.name: file_sha256(p) for p in args.output.iterdir()},
    }
    with (args.output / "build_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"audit": audit, "review_items": receipt["review_items"]}))


if __name__ == "__main__":
    main()
