# One frozen recovery of length-capped monitor forecasts

20 November 2026. This is a transparent **post-observation budget repair**, specified before any new 512-token forecast. The original [communication study](./COMMUNICATION_NEXT_STUDY_DESIGN.md), reports, readers, predictions and invalid outputs remain unchanged. This extension does not create an independently held-out confirmatory baseline or a new positive research claim.

The 160-token full-context forecaster has 66 length-terminated outputs. Its other 78 outputs reached EOS and satisfy the unchanged strict parser. A punctuation diagnostic changes none of those classifications. Therefore the recovery selects **every original `finish_reason == "length"` forecast**, independent of correctness, recipient outcome, sender objective, probe score or annotation. All 78 EOS forecasts are reused exactly.

## Frozen change and invariants

The sole generation change is `max_new_tokens: 160 → 512`. Keep the original greedy generation settings, seed 42, pinned Llama 3.1 8B Instruct revision, NF4 weights/BF16 compute, chat template and runtime identity. Reconstruct each prompt with the original `monitor_messages(case, original_row, saved_report)` helper. Use its original `monitor_decision` parser: an output still reaching the cap remains INVALID, and an EOS output needs exactly one matching FINAL line. No parser, prompt, report, recipient policy, factuality label, fitted weight or layer is changed.

Before model loading or new generation, the [runner](../scripts/extend_communication_monitor.py) writes an immutable protocol containing all 144 prompt/source bindings, all 66 selected identities, 78 cached identities, the selection rule, source-tree hashes, source-code hashes and exact old/new settings. Actual reader answers are not part of model jobs. They are joined only for evaluation after generation.

After loading the same model, reconstruct and compare **all 144 encoded prompt token-ID sequences** against their saved originals before generating the first new token. Generate the 66 selected forecasts from scratch under the larger cap. Before using any new forecast, require both its prompt IDs and **its entire original 160-token generated prefix** to match exactly. Checking just text, an initial handful of tokens, or a parsed answer is insufficient.

Save each attempted generation and its replay audit before accepting or rejecting it. A mismatch aborts the batch with an `inconclusive_replay_mismatch` manifest, preserves the attempt and provides no completed recovery summary. A runtime failure likewise preserves completed attempts. Do not silently fall back to a different prompt, precision, seed, checkpoint, model, parser or third budget. There is no automatic retry or resume. No new sender reports, recipient calls or probe fits are authorized by this protocol.

## Outputs and comparison

The new native WSL output is `/home/adirj/kcl-latent-probes-runs/communication_monitor_512_v1`. The original communication run is read-only. Preserve both runs.

On success publish an all-144 forecast registry with explicit cached-versus-extended provenance, all replay checks, unchanged original monitor predictions, recovered predictions and source record hashes. Report the original and recovered categorical accuracy, macro F1, class counts and per-class recall by sender goal and relevant case/context groups. Actual INVALID reader targets remain visible and excluded under the original metric rule; an INVALID forecast counts as a failed forecast on valid targets. Repeated recipient contexts are not independent cases.

Report original monitor costs and additional generation costs separately: number of calls, input tokens, generated tokens and measured generation seconds. The 66 new calls regenerate the first 160 tokens as well as any continuation, so disclose that full cost. Cached EOS forecasts require zero additional model calls. This recovery has a larger computational budget than the original monitor, and every comparison must say so.

This extension does not show superiority to exact reader simulation, establish a reader-belief mechanism, remove annotation uncertainty or validate a new deception label. A useful result is a properly budgeted text comparison whose remaining failures can be interpreted without conflating them with output truncation.
