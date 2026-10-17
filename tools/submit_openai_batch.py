#!/usr/bin/env python3
"""Submit batch request file(s) to the OpenAI Batch API, poll until done,
and download the result files.

For each request file <name>.jsonl this uploads it, creates a batch on
/v1/chat/completions with a 24h window, polls every --poll-interval seconds,
and writes the results to <name>_output.jsonl (plus <name>_errors.jsonl when
the batch has an error file). The batch ID is saved in
<name>_batch_id.txt, so an interrupted run resumes polling the existing batch
instead of paying for a duplicate submission. Already-downloaded outputs are
skipped.

Used by both the rollout stage (rollout/build_openai_batch.py) and the
scoring stage (scoring/score_batch.py phase1/phase2).

Example:
  python tools/submit_openai_batch.py \
      --requests batches/train_rollout/batch_*[0-9].jsonl
  python rollout/merge_openai_batch.py \
      --batch-output batches/train_rollout/*_output.jsonl \
      --out rollouts/gpt4o_train.jsonl
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", nargs="+", required=True,
                        help="request jsonl file(s) from build_openai_batch.py or "
                        "score_batch.py")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[args.api_key_env])

    # Submit one batch per request file (skip files whose output exists).
    pending: dict[str, Path] = {}  # batch_id -> request path
    for path_str in args.requests:
        path = Path(path_str)
        out_path = path.with_name(path.stem + "_output.jsonl")
        if out_path.exists():
            print(f"skip {path.name}: {out_path.name} already exists")
            continue
        batch_id_path = path.with_name(path.stem + "_batch_id.txt")
        if batch_id_path.exists():
            batch_id = batch_id_path.read_text(encoding="utf-8").strip()
            if not batch_id:
                raise SystemExit(f"empty saved batch ID: {batch_id_path}")
            pending[batch_id] = path
            print(f"resume {path.name} as batch {batch_id}")
            continue
        with open(path, "rb") as request_file:
            uploaded = client.files.create(file=request_file, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source_file": path.name},
        )
        pending[batch.id] = path
        batch_id_path.write_text(batch.id + "\n", encoding="utf-8")
        print(f"submitted {path.name} as batch {batch.id}")

    # Poll until every batch reaches a terminal status.
    while pending:
        time.sleep(args.poll_interval)
        for batch_id in list(pending):
            batch = client.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(
                f"{batch_id} [{pending[batch_id].name}]: {batch.status} "
                f"({counts.completed}/{counts.total} done, {counts.failed} failed)"
            )
            if batch.status not in TERMINAL_STATUSES:
                continue
            path = pending.pop(batch_id)
            if batch.output_file_id:
                out_path = path.with_name(path.stem + "_output.jsonl")
                out_path.write_bytes(client.files.content(batch.output_file_id).read())
                print(f"  wrote {out_path}")
            if batch.error_file_id:
                err_path = path.with_name(path.stem + "_errors.jsonl")
                err_path.write_bytes(client.files.content(batch.error_file_id).read())
                print(f"  wrote {err_path} (per-request errors)")
            if batch.status != "completed":
                batch_id_path = path.with_name(path.stem + "_batch_id.txt")
                print(
                    f"  WARNING: batch ended as {batch.status}; inspect the "
                    f"error file, then remove {batch_id_path.name} before "
                    "submitting a replacement"
                )

    print("All batches finished.")


if __name__ == "__main__":
    main()
