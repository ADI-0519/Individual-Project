# Audit source formatting and provenance refresh: 22 November 2026

This refresh makes the supplementary measurement/query helpers conform to the
repository's Ruff format policy while preserving the exact source bytes behind
the already-executed audits. It adds provenance metadata and safe output guards;
it does not change measurement definitions, source responses, model predictions,
gold labels or any numerical diagnostic result. No model inference or fitting
occurred.

## Historical evidence remains recoverable

Before editing any helper or test, the exact four current source files were
copied byte-for-byte to [source snapshots](validation/audit-source-snapshots-2026-11-22/)
with `.py.txt` extensions. The [snapshot index](validation/audit-source-snapshots-2026-11-22/index.json)
records each original absolute and repository-relative path, byte count,
SHA256, snapshot path and historical report links. Each copied digest was
checked against the original query audit's source/protected-file bindings.

The original [measurement Markdown](MAIN_STUDY_MEASUREMENT_FORENSICS_2026-11-22.md),
[measurement JSON](validation/2026-11-22-main-study-measurement-forensics.json),
[query Markdown](MAIN_STUDY_QUERY_ANCHORING_POSTHOC_2026-11-22.md) and
[query JSON](validation/2026-11-22-main-study-query-anchoring-posthoc.json) remain
unchanged. Their exact digests are also indexed. Source hashes printed in those
historical reports now identify the corresponding immutable snapshots, rather
than the formatted current paths. No historical binding is silently reassigned
to different bytes.

The snapshots are evidence, not runnable current code. Do not format or edit
them. Their preservation is covered by the current query tests.

## Current sources and new v2 artifacts

The current [measurement helper](../scripts/audit_main_study_measurement_forensics.py)
and [measurement tests](../tests/test_main_study_measurement_forensics.py), plus
the current [query helper](../scripts/audit_main_study_query_anchoring.py) and
[query tests](../tests/test_main_study_query_anchoring.py), pass Ruff lint and
format checks. Both helpers refuse to overwrite an existing artifact and write
fresh output with fixed LF newlines. Their default output names are now v2.

Fresh runs produced [measurement v2](validation/2026-11-22-main-study-measurement-forensics-v2.json)
and [query v2](validation/2026-11-22-main-study-query-anchoring-posthoc-v2.json).
The v2 files bind current code/tests, their actual inputs, the immutable
historical reports and the snapshot index. Query v2 binds the **current**
measurement helper/tests but defaults to the **original measurement JSON** and
unchanged measurement Markdown. Selecting a different measurement JSON requires
the explicit `--measurement-audit-report` argument; no implicit substitution
of measurement v2 occurs.

| Artifact | Current SHA256 |
| --- | --- |
| Measurement helper | `029aa397dcea7674be96433b9f4fcbc99f5bf6262e02252efb5cf3ef2b6f220f` |
| Measurement tests | `1d68dc2773932f956fed3d8c5872ea9f54e0a4ac9a69c6f7116ef40b8dbf5627` |
| Query helper | `c30bfd9fba2ccc25779d654a8732001a14d9841597cd8a1e14c7b65133d753db` |
| Query tests | `84cd7eb822d0b892ad7ac42c518bd716e4484d858e0940d89052a13a1f8cdef2` |
| Measurement v2 JSON | `f4c965b8604b8c4ae161acaec5a7cb22131f7962009a9b4661fae1b15280673d` |
| Query v2 JSON | `715b90b01be87d5fde3ddf383f591458ec65b3d41021517ceec74d3a3f2ca56e` |
| Historical snapshot index | `8b388fc3e8a6c5fbe9d8f56ef9209e7bd6070753c395b90e427ce24224f484e2` |

Every binding has a logical root (`workspace`, `run` or `fits`) and a relative
POSIX path, in addition to its recorded native path and digest. The query
helper's verification function accepts a root mapping, so Windows, WSL or a
relocated workspace can verify the same evidence without interpreting a foreign
absolute path. It rejects path traversal, drive-qualified relative paths and
hash changes. Recorded native paths are provenance, not the portable locator.

## Validation

All **23 relevant tests pass**, including the original measurement/query
failure cases, historical snapshot bindings, both current v2 workspace
bindings, stale-native-path relocation and hash-corruption rejection. All
**13 measurement** and **14 query** source bindings were additionally verified
with the live workspace/run/fit root mapping.

Numerical equality with v1 was checked directly: measurement models, geometry,
monitor diagnostics, token summaries, source-fold mismatch results and bootstrap
counterexample are identical; query overall, mechanism, group and every row's
result are identical. The correction same-response result therefore remains
81 → 90 correct quartets out of 240, with all 44 original invalid responses
retained across mechanisms. These remain posthoc diagnostics.

The following commands ran from the project root after configuring its known
inner virtual environment and setting `PYTHONPATH` to the project source:

```powershell
& ./.venv/Scripts/python.exe -m ruff format scripts/audit_main_study_measurement_forensics.py tests/test_main_study_measurement_forensics.py scripts/audit_main_study_query_anchoring.py tests/test_main_study_query_anchoring.py
& ./.venv/Scripts/python.exe -m ruff check scripts/audit_main_study_measurement_forensics.py tests/test_main_study_measurement_forensics.py scripts/audit_main_study_query_anchoring.py tests/test_main_study_query_anchoring.py
& ./.venv/Scripts/python.exe scripts/audit_main_study_measurement_forensics.py --output docs/validation/2026-11-22-main-study-measurement-forensics-v2.json
& ./.venv/Scripts/python.exe scripts/audit_main_study_query_anchoring.py --output docs/validation/2026-11-22-main-study-query-anchoring-posthoc-v2.json
& ./.venv/Scripts/python.exe -m pytest -q tests/test_main_study_measurement_forensics.py tests/test_main_study_query_anchoring.py
& ./.venv/Scripts/python.exe -m ruff format --check scripts/audit_main_study_measurement_forensics.py tests/test_main_study_measurement_forensics.py scripts/audit_main_study_query_anchoring.py tests/test_main_study_query_anchoring.py
```

Re-execution requires a fresh output name; existing v1/v2 evidence is protected
by the output guard. The query v2 file preserves all 3,840 row traces and is
approximately 3.1 MB, so it belongs with the existing derived query-evidence
exception to the repository's generic large-file hook. This refresh did not
edit that configuration, the cached fitting pipeline, its protocol, or the
original frozen study.
