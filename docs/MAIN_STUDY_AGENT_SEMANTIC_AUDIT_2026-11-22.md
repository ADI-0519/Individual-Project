# Main-study agent semantic audit ? 2026-11-22

This is an independent **AGENT** audit. It is not human review and supplies no human acceptance or approval.

## Finding

All 736 representative prompts admit a definite answer to the question actually asked. The independent answers are **274 YES, 462 NO, 0 UNSURE**. Across all 154 narrative frames and every correction path, no genuine ambiguity in the requested sender-belief-about-reader judgment, internal contradiction, or prose defect that prevents assigning that judgment was found. There are **0 unresolved semantic blockers within this reviewed packet**. Formal gold-label agreement has not been measured in this blind artifact.

Actual private reader belief is a different quantity. The prose explicitly withholds whether accepted reading/access records agree with actual private access. Actual reader location belief is underdetermined in **488/736** representative prompts. In the remaining **248/736**, initial and possible newly acquired evidence give the same location, so the actual belief is fixed despite undisclosed access. This does not make any of the 736 requested second-order answers ambiguous.

## Independence and bindings

- Source: [prompt-only review packet](review/main_study_v1/review_prompts.jsonl).
- Raw review-packet SHA256: `fa4791f72c8fb2743d5d29361ea38f8b1ad0e94cd3795250f189f28735f97751`.
- Canonical bank SHA256, externally supplied metadata only: `e038025571a38a7f2c489e5f8c3bf3b0a3c11a05f9b8c7e89d26292a4e7bfb1d`.
- Independent answers: [agent annotation JSON](../data/annotations/main_study_v1_agent_semantic_20261122.json).
- Frozen answer-artifact SHA256: `71b021952823a65ba345763100c8d8a493e16363ce7317dc068fba429ee028af`.
- Created: `2026-11-22T13:25:06.784941Z`.

Only the packet schema and system/user/query prose were used to answer. The packet contains case ID, group ID, supplied prompt hash, and messages; it contains no formal gold label or state field. IDs were treated as opaque keys. No design implementation, bank source, model result, or previous semantic audit was opened. No model/GPU/API evaluation was run. No bank, model code, or human ledger was changed.

## Exact coverage and method

All **154** distinct narrative openings and their channel/session setup descriptions were read, including each setting, role assignment, and any restrictions on alternate evidence. Every one of the **736** complete prompt bodies was checked by exact deterministic normalization. Normalization removes only the already-inspected opening/setup and consistently replaces the two names and two concrete location names with placeholders; it retains the full evidence chronology, accepted records, update rules, privacy clauses, candidate, and final query. This produced exactly **16 complete normalized semantic paths**, all of which were read in full. No residual narrative-body variant was omitted.

| Mechanism | Frames | Representative prompts | Complete paths per frame | YES | NO |
|---|---:|---:|---:|---:|---:|
| Single message | 62 | 248 | 4 | 62 | 186 |
| Direct observation | 62 | 248 | 4 | 62 | 186 |
| Replacement correction | 30 | 240 | 8 | 150 | 90 |
| Total | 154 | 736 | ? | 274 | 462 |

This is grouped agent reading plus a prose-derived deterministic answer assignment, not 736 separately elicited human judgments. Each row of the JSON includes its literal names, initial location, report/session location, accepted record wording, replacement wording where present, private world location, queried location, attributed reader location, and resulting answer. The JSON also contains the full normalized path texts/hashes and an explicit inventory of all frames and their cases.

## Literal reasoning

The final query asks whether the sender/supervisor thinks that the reader believes the candidate proposition. The jointly witnessed 09:00 inspection establishes the reader?s initial belief and the sender?s knowledge of it. A reading/access record is supplied to the sender and explicitly accepted as definitive. It is evidence for the sender?s view of the reader; it is not a narrator guarantee about actual private access.

For a single message, the sender attributes the report location if the accepted record says it was fully read; otherwise the sender attributes the initial location. The report is an ordinary claim, with explicit trust even when false. For direct observation, the sender attributes the displayed location if the accepted record says the complete session was observed; otherwise the sender attributes the initial location. The image supplies the displayed location to the sender without exposing the audience.

For corrections, a fully read replacement supplies an explicit new location and takes priority even when the first report was not read. If the replacement is recorded as unread, a read first report still determines the attributed belief; if neither report is recorded as read, the initial belief remains. The replacement cannot naturally be interpreted as a bare withdrawal: the prose expressly supplies the new location and states that it is not merely withdrawing the earlier report.

The sender?s accurate private 09:15 check determines their first-order world knowledge. The reader cannot access the check or receive notice of it, gets no extra location evidence, and cannot infer unseen movements. The later check therefore does not silently update the reader belief attributed by the sender. The candidate is expressly a statement to assess, not new evidence or a received message.

## All correction paths

Let R be the first report location and C the replacement/candidate location. Each row below occurs in all 30 correction frames.

| Initial location | First recorded read | Replacement recorded read | Attributed reader location | Answer for C | Cases |
|---|---|---|---|---|---:|
| R | No | No | R | NO | 30 |
| R | No | Yes | C | YES | 30 |
| R | Yes | No | R | NO | 30 |
| R | Yes | Yes | C | YES | 30 |
| C | No | No | C | YES | 30 |
| C | No | Yes | C | YES | 30 |
| C | Yes | No | R | NO | 30 |
| C | Yes | Yes | C | YES | 30 |

## Ambiguity, assumptions, and limits

No opening/setup creates an additional route for learning location that conflicts with the explicit later evidence restrictions. Different occupational settings, token stand-ins, cameras, registers, and administrative roles do not override the stated evidence and update rules. The only ordinary-language assumptions are that the two differently named containers denote distinct locations and that the singular token has one location at a time. No case-specific unstated belief-update or communications assumption was needed.

The undisclosed truth of private access is deliberate missing information about the actual reader, not a missing premise for the stated second-order target. Changing the task to ask what the reader actually believes would require more access evidence in the 488 underdetermined cases. No such changed task is used here.

World-location independence and consistent global exchanges of names/locations were reviewed at the prose level. A global renaming preserves the relations in the evidence and query; changing the candidate alone between the two possible believed locations complements the answer. A private world change supplies no new reader evidence. Expanded bank prompts outside the review packet were not inspected individually, so this report does not claim a full-bank rendered-text inspection.

The supplied per-prompt hashes are preserved as identifiers. The raw packet hash independently binds the reviewed bytes; producer hash serialization was not inferred. The bank digest was received as opaque metadata, not obtained by opening the bank. Any comparison to formal labels must happen after this frozen independent artifact; no such comparison is claimed here.

## Frame inventory

All rows below had their unique narrative opening/setup inspected and all indicated complete paths covered. No frame-specific blocker was found.

| # | Opaque frame ID | Mechanism | Prompts | YES | NO |
|---:|---|---|---:|---:|---:|
| 1 | `train_message_00a` | message | 4 | 1 | 3 |
| 2 | `train_message_00b` | message | 4 | 1 | 3 |
| 3 | `train_message_01a` | message | 4 | 1 | 3 |
| 4 | `train_message_01b` | message | 4 | 1 | 3 |
| 5 | `train_message_02a` | message | 4 | 1 | 3 |
| 6 | `train_message_02b` | message | 4 | 1 | 3 |
| 7 | `train_message_03a` | message | 4 | 1 | 3 |
| 8 | `train_message_03b` | message | 4 | 1 | 3 |
| 9 | `train_message_04a` | message | 4 | 1 | 3 |
| 10 | `train_message_04b` | message | 4 | 1 | 3 |
| 11 | `train_message_05a` | message | 4 | 1 | 3 |
| 12 | `train_message_05b` | message | 4 | 1 | 3 |
| 13 | `train_message_06a` | message | 4 | 1 | 3 |
| 14 | `train_message_06b` | message | 4 | 1 | 3 |
| 15 | `train_message_07a` | message | 4 | 1 | 3 |
| 16 | `train_message_07b` | message | 4 | 1 | 3 |
| 17 | `train_message_08a` | message | 4 | 1 | 3 |
| 18 | `train_message_08b` | message | 4 | 1 | 3 |
| 19 | `train_message_09a` | message | 4 | 1 | 3 |
| 20 | `train_message_09b` | message | 4 | 1 | 3 |
| 21 | `train_message_10a` | message | 4 | 1 | 3 |
| 22 | `train_message_10b` | message | 4 | 1 | 3 |
| 23 | `train_message_11a` | message | 4 | 1 | 3 |
| 24 | `train_message_11b` | message | 4 | 1 | 3 |
| 25 | `train_observation_00a` | observation | 4 | 1 | 3 |
| 26 | `train_observation_00b` | observation | 4 | 1 | 3 |
| 27 | `train_observation_01a` | observation | 4 | 1 | 3 |
| 28 | `train_observation_01b` | observation | 4 | 1 | 3 |
| 29 | `train_observation_02a` | observation | 4 | 1 | 3 |
| 30 | `train_observation_02b` | observation | 4 | 1 | 3 |
| 31 | `train_observation_03a` | observation | 4 | 1 | 3 |
| 32 | `train_observation_03b` | observation | 4 | 1 | 3 |
| 33 | `train_observation_04a` | observation | 4 | 1 | 3 |
| 34 | `train_observation_04b` | observation | 4 | 1 | 3 |
| 35 | `train_observation_05a` | observation | 4 | 1 | 3 |
| 36 | `train_observation_05b` | observation | 4 | 1 | 3 |
| 37 | `train_observation_06a` | observation | 4 | 1 | 3 |
| 38 | `train_observation_06b` | observation | 4 | 1 | 3 |
| 39 | `train_observation_07a` | observation | 4 | 1 | 3 |
| 40 | `train_observation_07b` | observation | 4 | 1 | 3 |
| 41 | `train_observation_08a` | observation | 4 | 1 | 3 |
| 42 | `train_observation_08b` | observation | 4 | 1 | 3 |
| 43 | `train_observation_09a` | observation | 4 | 1 | 3 |
| 44 | `train_observation_09b` | observation | 4 | 1 | 3 |
| 45 | `train_observation_10a` | observation | 4 | 1 | 3 |
| 46 | `train_observation_10b` | observation | 4 | 1 | 3 |
| 47 | `train_observation_11a` | observation | 4 | 1 | 3 |
| 48 | `train_observation_11b` | observation | 4 | 1 | 3 |
| 49 | `dev_message_01` | message | 4 | 1 | 3 |
| 50 | `dev_message_02` | message | 4 | 1 | 3 |
| 51 | `dev_message_03` | message | 4 | 1 | 3 |
| 52 | `dev_message_04` | message | 4 | 1 | 3 |
| 53 | `dev_message_05` | message | 4 | 1 | 3 |
| 54 | `dev_message_06` | message | 4 | 1 | 3 |
| 55 | `dev_message_07` | message | 4 | 1 | 3 |
| 56 | `dev_message_08` | message | 4 | 1 | 3 |
| 57 | `dev_observation_01` | observation | 4 | 1 | 3 |
| 58 | `dev_observation_02` | observation | 4 | 1 | 3 |
| 59 | `dev_observation_03` | observation | 4 | 1 | 3 |
| 60 | `dev_observation_04` | observation | 4 | 1 | 3 |
| 61 | `dev_observation_05` | observation | 4 | 1 | 3 |
| 62 | `dev_observation_06` | observation | 4 | 1 | 3 |
| 63 | `dev_observation_07` | observation | 4 | 1 | 3 |
| 64 | `dev_observation_08` | observation | 4 | 1 | 3 |
| 65 | `test_correction_01` | correction | 8 | 5 | 3 |
| 66 | `test_correction_02` | correction | 8 | 5 | 3 |
| 67 | `test_correction_03` | correction | 8 | 5 | 3 |
| 68 | `test_correction_04` | correction | 8 | 5 | 3 |
| 69 | `test_correction_05` | correction | 8 | 5 | 3 |
| 70 | `test_correction_06` | correction | 8 | 5 | 3 |
| 71 | `test_correction_07` | correction | 8 | 5 | 3 |
| 72 | `test_correction_08` | correction | 8 | 5 | 3 |
| 73 | `test_correction_09` | correction | 8 | 5 | 3 |
| 74 | `test_correction_10` | correction | 8 | 5 | 3 |
| 75 | `test_correction_11` | correction | 8 | 5 | 3 |
| 76 | `test_correction_12` | correction | 8 | 5 | 3 |
| 77 | `test_correction_13` | correction | 8 | 5 | 3 |
| 78 | `test_correction_14` | correction | 8 | 5 | 3 |
| 79 | `test_correction_15` | correction | 8 | 5 | 3 |
| 80 | `test_correction_16` | correction | 8 | 5 | 3 |
| 81 | `test_correction_17` | correction | 8 | 5 | 3 |
| 82 | `test_correction_18` | correction | 8 | 5 | 3 |
| 83 | `test_correction_19` | correction | 8 | 5 | 3 |
| 84 | `test_correction_20` | correction | 8 | 5 | 3 |
| 85 | `test_correction_21` | correction | 8 | 5 | 3 |
| 86 | `test_correction_22` | correction | 8 | 5 | 3 |
| 87 | `test_correction_23` | correction | 8 | 5 | 3 |
| 88 | `test_correction_24` | correction | 8 | 5 | 3 |
| 89 | `test_correction_25` | correction | 8 | 5 | 3 |
| 90 | `test_correction_26` | correction | 8 | 5 | 3 |
| 91 | `test_correction_27` | correction | 8 | 5 | 3 |
| 92 | `test_correction_28` | correction | 8 | 5 | 3 |
| 93 | `test_correction_29` | correction | 8 | 5 | 3 |
| 94 | `test_correction_30` | correction | 8 | 5 | 3 |
| 95 | `test_message_01` | message | 4 | 1 | 3 |
| 96 | `test_message_02` | message | 4 | 1 | 3 |
| 97 | `test_message_03` | message | 4 | 1 | 3 |
| 98 | `test_message_04` | message | 4 | 1 | 3 |
| 99 | `test_message_05` | message | 4 | 1 | 3 |
| 100 | `test_message_06` | message | 4 | 1 | 3 |
| 101 | `test_message_07` | message | 4 | 1 | 3 |
| 102 | `test_message_08` | message | 4 | 1 | 3 |
| 103 | `test_message_09` | message | 4 | 1 | 3 |
| 104 | `test_message_10` | message | 4 | 1 | 3 |
| 105 | `test_message_11` | message | 4 | 1 | 3 |
| 106 | `test_message_12` | message | 4 | 1 | 3 |
| 107 | `test_message_13` | message | 4 | 1 | 3 |
| 108 | `test_message_14` | message | 4 | 1 | 3 |
| 109 | `test_message_15` | message | 4 | 1 | 3 |
| 110 | `test_message_16` | message | 4 | 1 | 3 |
| 111 | `test_message_17` | message | 4 | 1 | 3 |
| 112 | `test_message_18` | message | 4 | 1 | 3 |
| 113 | `test_message_19` | message | 4 | 1 | 3 |
| 114 | `test_message_20` | message | 4 | 1 | 3 |
| 115 | `test_message_21` | message | 4 | 1 | 3 |
| 116 | `test_message_22` | message | 4 | 1 | 3 |
| 117 | `test_message_23` | message | 4 | 1 | 3 |
| 118 | `test_message_24` | message | 4 | 1 | 3 |
| 119 | `test_message_25` | message | 4 | 1 | 3 |
| 120 | `test_message_26` | message | 4 | 1 | 3 |
| 121 | `test_message_27` | message | 4 | 1 | 3 |
| 122 | `test_message_28` | message | 4 | 1 | 3 |
| 123 | `test_message_29` | message | 4 | 1 | 3 |
| 124 | `test_message_30` | message | 4 | 1 | 3 |
| 125 | `test_observation_01` | observation | 4 | 1 | 3 |
| 126 | `test_observation_02` | observation | 4 | 1 | 3 |
| 127 | `test_observation_03` | observation | 4 | 1 | 3 |
| 128 | `test_observation_04` | observation | 4 | 1 | 3 |
| 129 | `test_observation_05` | observation | 4 | 1 | 3 |
| 130 | `test_observation_06` | observation | 4 | 1 | 3 |
| 131 | `test_observation_07` | observation | 4 | 1 | 3 |
| 132 | `test_observation_08` | observation | 4 | 1 | 3 |
| 133 | `test_observation_09` | observation | 4 | 1 | 3 |
| 134 | `test_observation_10` | observation | 4 | 1 | 3 |
| 135 | `test_observation_11` | observation | 4 | 1 | 3 |
| 136 | `test_observation_12` | observation | 4 | 1 | 3 |
| 137 | `test_observation_13` | observation | 4 | 1 | 3 |
| 138 | `test_observation_14` | observation | 4 | 1 | 3 |
| 139 | `test_observation_15` | observation | 4 | 1 | 3 |
| 140 | `test_observation_16` | observation | 4 | 1 | 3 |
| 141 | `test_observation_17` | observation | 4 | 1 | 3 |
| 142 | `test_observation_18` | observation | 4 | 1 | 3 |
| 143 | `test_observation_19` | observation | 4 | 1 | 3 |
| 144 | `test_observation_20` | observation | 4 | 1 | 3 |
| 145 | `test_observation_21` | observation | 4 | 1 | 3 |
| 146 | `test_observation_22` | observation | 4 | 1 | 3 |
| 147 | `test_observation_23` | observation | 4 | 1 | 3 |
| 148 | `test_observation_24` | observation | 4 | 1 | 3 |
| 149 | `test_observation_25` | observation | 4 | 1 | 3 |
| 150 | `test_observation_26` | observation | 4 | 1 | 3 |
| 151 | `test_observation_27` | observation | 4 | 1 | 3 |
| 152 | `test_observation_28` | observation | 4 | 1 | 3 |
| 153 | `test_observation_29` | observation | 4 | 1 | 3 |
| 154 | `test_observation_30` | observation | 4 | 1 | 3 |
