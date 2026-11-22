# Fresh test-context authoring v2

This agent-authored draft contains 24 held-out test contexts in
[authored_test_contexts_v1.json](../data/belief_main/fresh_readout_v2/authored_test_contexts_v1.json).
It is `draft_unreviewed`, with no human approval and zero model observations.
No model/API calls, model predictions, native-case outcomes or new literature
claims informed this authoring. The author read the complete
[draft study specification](FRESH_READOUT_STUDY_SPECIFICATION_V2.md) and used only
the authored context/entity fields of the
[old bank](../data/belief_main/main_v1/main_bank_v1.json) for overlap avoidance.

Each context supplies roles, two physical storage locations, a token standing
for an object in a harmless fictional workflow, and the availability of a
private written channel. It does not supply symbolic factors, timestamps,
report contents, a report count, reading events, nested-belief labels or world
values. A later private physical check is compatible with every setting. One-step
reports, easy correction, hard correction and explicit final attribution must
be supplied by the two fixed renderers. Context prose does not make a report
true, imply that the recipient reads it, or communicate the private world check.

The workflow coverage is deliberately varied: claim matching, editorial query
routing, software issue triage, sports rotation planning, game scenario balance,
animation timing revision, mock ballot handling, office layout planning, price
review, prop-reset preparation, packaging folds, exchange-offer cataloguing,
fragrance naming, adjacent campsite booking, household duty scheduling, speech
sequencing, appointment spacing, shelf-label alignment, repair prioritization,
lighting-cue alignment, merge-field approval, inscription line breaks, scorecard
sequence proofing and jigsaw cut-pattern review. Related broad settings can
still share vocabulary or administrative structure with old material; different
workflow purposes and represented objects are authoring choices, not proof of
semantic independence.

Actor names are conventional two-word ASCII full names, reserved with the parent
author before freezing the records. Candidate locations use ordinary two- to four-word noun
phrases. Locations are containers, so a drawn plan, game board, programme,
software display, completed product or public exhibit cannot silently serve as
a second view of the token. Field-specific notes identify possible alternative
channels and explain the intended workflow distinctions; those notes are audit
metadata and should not be inserted into the rendered task.

The author self-reviewed all 24 base contexts for workflow distinctions,
channel compatibility and unintended token observations. This is an authoring
check, not the planned author-independent interpretation of rendered tasks.

Mechanical checks cover the required IDs, test split, exact record fields,
unique wording groups, two distinct locations per context, actor-component and
full-location overlap against all 154 old groups, formatting placeholders,
normalized framing duplicates, word five-grams against old framing, and UTF-8
without a byte-order mark with LF line endings. These checks passed: 96 distinct
actor-name components and 48 distinct location strings have no corresponding
old-bank reuse; there are no normalized framing duplicates and no normalized
word five-gram matches against the old introduction-plus-event framing. The
five-gram check lowercases and retains alphanumeric word tokens, including the
placeholder names; it is a surface comparison. Cross-cohort entity checks and
author-independent interpretation of base contexts and rendered samples remain
the parent workflow's responsibility. Surface overlap checks do not establish
semantic equivalence or independence, and this draft claims no human review,
validated model behavior or guaranteed experimental result.
