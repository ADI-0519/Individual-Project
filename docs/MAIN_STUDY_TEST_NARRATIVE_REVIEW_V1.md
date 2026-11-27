# Independent agent review of the draft test narratives

21 November 2026. This is a review by a different authoring agent, not an
independent human review, participant annotation or completed semantic approval.
The human review ledger remains blank. No model outputs informed this review.

## Material read

Read the introductions and event descriptions of all 90 groups in
[the test authoring source](../data/belief_main/authored_test_v1.json):
30 observation, 30 message and 30 correction contextual framings. Read the
[shared renderer](../src/latent_probes/main_study_bank.py) and the
[study specification](MAIN_STUDY_SPECIFICATION_V1.md) before assessing them.

Also read every complete expanded representative account for correction groups
03 (jazz festival), 16 (moss exhibition), and 24 (architecture society): eight
paths per group at first-report slot0, later-world slot0, canonical name map.
This covers initial matches/does not match the first report crossed with both
accepted reading flags. It does not constitute reading all 3,840 expanded test
prefixes, nor establish semantic independence of all 90 contexts.

Reviewed test-source raw SHA256:
`d28de9deeac15fb2d66fb72e95786b99eb3325bdd01df78e85545299511a5d4b`.
Later authoring edits invalidate this exact-byte binding and need a recorded
follow-up check; they do not silently inherit this review.

## Semantic findings

No case-specific contradiction was found between the reviewed introductory/event
prose and the crossed state construction. Observation descriptions provide a
visual check without asserting reader attendance. A photographed or filmed
check is compatible with the renderer's directly observed session; the sender's
image excludes the reader area. Initial observation and later private world
checks remain distinct.

The message introductions do not assert that their report is true or read.
Phrases such as "prepares a report" are preparation descriptions followed by the
renderer's explicit sending event, not additional communicated reports. The
correction descriptions introduce a reporting exercise without supplying an
extra location, reading receipt or correction outcome. Their anticipation of
revisable messages is a mechanism cue, not a target label.

The three fully expanded correction groups preserve the required truth table.
Writing M1=0 and M2=1, the expected attributed location is:

| Initial | First read B1 | Replacement read B2 | Attributed location |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 1 | 1 |

The sender's private later check does not update the attributed reader belief.
The candidate proposition is expressly withheld from the fictional characters.
The final-report text supplies a replacement location, avoiding the unknown
state that a bare retraction would introduce.

## Limitations and changes worth considering before freeze

1. **Name disjointness requires a precise definition.** The training/development
   author uses two-part names; the test source uses single first names. Exact
   actor strings are different, but some name tokens overlap, for example Inez,
   Willem, Gethin, Farid, Giselle, Umar, Cerys and Pascal. This does not disclose
   gold, but an exact-string overlap check cannot support a claim that all actor
   name tokens are unseen. Use a consistent actor-naming policy or report the
   narrower exact-string condition.
2. **Authoring style varies with partition and mechanism.** Test stories are
   predominantly administrative rehearsals, whereas training includes more
   equipment-use lessons. Correction locations often have two colour words;
   one-step locations often have one. Correction prose also mentions revision
   before the explicit reports. Together with the extra event and longer
   timeline, these features limit attribution of transfer difficulty to pure
   logical composition. The two fitting arms share each exact test prompt, and
   the I=M1 secondary correction cells share the correction format, which help
   interpretation but do not eliminate this limitation.
3. **Ninety framings are not ninety reasoning families.** Nearly all test items
   concern mock registers, allocation cards or inventories. Their proper names
   and domains vary, while the shared renderer supplies the decisive evidence
   and the three logical mechanisms. Group-level uncertainty remains conditional
   on this authored distribution; do not imply naturalistic deception coverage.
4. **Contextual domains recur without verbatim reuse.** Museum, botanical,
   theatre-related and library contexts overlap broad domains of the excluded
   design packet. The reviewed test scenes are different tasks rather than
   copies of those exact six examples. Exclusion should be described in terms
   of the exact narratives/entities/wording groups actually checked, not as
   leaving out entire occupational domains.

These are pre-output design and reporting issues. They are not reasons to
discard difficult test cases after observing scores or to claim human review.
Any narrative edits require rebuilding hashes and regenerating the blank human
review packet before research execution.
