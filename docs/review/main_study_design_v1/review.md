# Main-study design examples: blinded semantic review

**Unexecuted design material, 21 November 2026.** No model outputs have been collected. All 24 examples and their six narrative groups are permanently excluded from future fitting, development, model selection, pilot and test datasets. Agent drafting does not count as human review.

This packet asks whether the prose makes each fictional perspective unambiguous. It deliberately omits state bits, gold answers, pair labels and any model results. There are six underlying narratives with four variants each; repeated variants are not independent scenarios. The order is fixed for review convenience, not a randomized test. The separate [annotations.csv](./annotations.csv) begins entirely blank in every annotation field.

Read the shared system text and each complete user prompt. Independently answer its question, then record the sender's attributed reader location, any ambiguity, contradictory timing, unintended information access or wording leakage. A missing message notification must not reveal a move. Later replacement reports must specify a new location rather than merely retracting a claim. Record uncertainty rather than guessing. Suggested ambiguity values: CLEAR, UNCLEAR, CONTRADICTORY; none have been filled in advance.

This is sender-access attribution: what the sender concludes from accepted records. It is not a question about inaccessible actual delivery or the actual private reader state. The sender knows the initial check, outgoing update content, trusted records and private current-world view. The question follows the original fixed slot-b nested-question wording. Actual-reader behavior, deception and spontaneous anticipation are not measured.

Each reviewer should use their own copy of the annotation table. The CSV has no preset answers or judgments. A later reconciliation must retain independent responses and disagreements. Neither agreement by software agents nor successful rule checks establish human semantic validation.

## Shared system message

Read the fictional scene. Questions will follow. Each person has only the observations and messages explicitly available to them. People remember the last place they observed the token and update after seeing a move or receiving a trusted message about its current place. They do not infer an unobserved move. Narrator-only information is not shared with a character merely because the narrator describes it.

## Exact user prompts

### design_001

At the museum conservation bench, Xiomara and Euan jointly inspect the token at 09:00. It is inside the copper drawer. They watch each other make this inspection. Euan believes the token is in the copper drawer, and Xiomara knows that initial belief.

At 09:05, an inspection port displays the token in the pearl drawer. Xiomara sees an accurate photograph from that session showing this location. The photograph excludes the viewing area, so it gives Xiomara no information about whether Euan looked through the port.

At 09:07, a session record supplied to Xiomara states: "Euan saw no part of the 09:05 inspection and received no notification that it occurred." Xiomara accepts this record as definitive and has no other evidence about Euan's participation.

Euan replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Euan gets no further sight of the token, message, movement notice or other location evidence. Xiomara knows these rules.

At 09:15, Xiomara alone makes an accurate live camera check: the token is now in the copper drawer. Euan cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Euan's view, with no movement notice available to Euan. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Xiomara retains the accepted records described above; Euan does not infer an unobserved move.

Candidate proposition: "The token is in the pearl drawer." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Xiomara think that Euan believes that the token is in the pearl drawer? Answer only YES or NO.

### design_002

At the museum conservation bench, Xiomara and Euan jointly inspect the token at 09:00. It is inside the copper drawer. They watch each other make this inspection. Euan believes the token is in the copper drawer, and Xiomara knows that initial belief.

At 09:05, an inspection port displays the token in the pearl drawer. Xiomara sees an accurate photograph from that session showing this location. The photograph excludes the viewing area, so it gives Xiomara no information about whether Euan looked through the port.

At 09:07, a session record supplied to Xiomara states: "Euan watched the complete 09:05 inspection through the port and saw the token in the pearl drawer." Xiomara accepts this record as definitive and has no other evidence about Euan's participation.

Euan replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Euan gets no further sight of the token, message, movement notice or other location evidence. Xiomara knows these rules.

At 09:15, Xiomara alone makes an accurate live camera check: the token is now in the copper drawer. Euan cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Euan's view, with no movement notice available to Euan. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Xiomara retains the accepted records described above; Euan does not infer an unobserved move.

Candidate proposition: "The token is in the pearl drawer." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Xiomara think that Euan believes that the token is in the pearl drawer? Answer only YES or NO.

### design_003

At the museum conservation bench, Xiomara and Euan jointly inspect the token at 09:00. It is inside the pearl drawer. They watch each other make this inspection. Euan believes the token is in the pearl drawer, and Xiomara knows that initial belief.

At 09:05, an inspection port displays the token in the pearl drawer. Xiomara sees an accurate photograph from that session showing this location. The photograph excludes the viewing area, so it gives Xiomara no information about whether Euan looked through the port.

At 09:07, a session record supplied to Xiomara states: "Euan saw no part of the 09:05 inspection and received no notification that it occurred." Xiomara accepts this record as definitive and has no other evidence about Euan's participation.

Euan replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Euan gets no further sight of the token, message, movement notice or other location evidence. Xiomara knows these rules.

At 09:15, Xiomara alone makes an accurate live camera check: the token is now in the copper drawer. Euan cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Euan's view, with no movement notice available to Euan. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Xiomara retains the accepted records described above; Euan does not infer an unobserved move.

Candidate proposition: "The token is in the pearl drawer." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Xiomara think that Euan believes that the token is in the pearl drawer? Answer only YES or NO.

### design_004

At the museum conservation bench, Xiomara and Euan jointly inspect the token at 09:00. It is inside the pearl drawer. They watch each other make this inspection. Euan believes the token is in the pearl drawer, and Xiomara knows that initial belief.

At 09:05, an inspection port displays the token in the pearl drawer. Xiomara sees an accurate photograph from that session showing this location. The photograph excludes the viewing area, so it gives Xiomara no information about whether Euan looked through the port.

At 09:07, a session record supplied to Xiomara states: "Euan watched the complete 09:05 inspection through the port and saw the token in the pearl drawer." Xiomara accepts this record as definitive and has no other evidence about Euan's participation.

Euan replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Euan gets no further sight of the token, message, movement notice or other location evidence. Xiomara knows these rules.

At 09:15, Xiomara alone makes an accurate live camera check: the token is now in the copper drawer. Euan cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Euan's view, with no movement notice available to Euan. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Xiomara retains the accepted records described above; Euan does not infer an unobserved move.

Candidate proposition: "The token is in the pearl drawer." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Xiomara think that Euan believes that the token is in the pearl drawer? Answer only YES or NO.

### design_005

At the coastal field station's 09:00 inventory, Thabo opens the basalt locker while Keiko watches. Both see the token there, and each sees the other looking. Keiko knows that Thabo's initial location belief is the basalt locker.

A visual inventory takes place at 09:05. Its accurate image shows the token inside the quartz locker. Keiko knows the image's contents from a separate copy, but that copy never shows the observers and cannot reveal whether Thabo attended.

The attendance entry that reaches Keiko at 09:07 reads: "Thabo did not view any of the 09:05 inventory and learned nothing about its occurrence." Keiko treats the entry as conclusive; no independent attendance evidence is available to Keiko.

Thabo replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Thabo gets no further sight of the token, message, movement notice or other location evidence. Keiko knows these rules.

At 09:15, Keiko alone makes an accurate live camera check: the token is now in the basalt locker. Thabo cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Thabo's view, with no movement notice available to Thabo. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Keiko retains the accepted records described above; Thabo does not infer an unobserved move.

Candidate proposition: "The token is in the basalt locker." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Keiko think that Thabo believes that the token is in the basalt locker? Answer only YES or NO.

### design_006

At the coastal field station's 09:00 inventory, Thabo opens the basalt locker while Keiko watches. Both see the token there, and each sees the other looking. Keiko knows that Thabo's initial location belief is the basalt locker.

A visual inventory takes place at 09:05. Its accurate image shows the token inside the quartz locker. Keiko knows the image's contents from a separate copy, but that copy never shows the observers and cannot reveal whether Thabo attended.

The attendance entry that reaches Keiko at 09:07 reads: "Thabo viewed the entire 09:05 visual inventory and directly saw the token in the quartz locker." Keiko treats the entry as conclusive; no independent attendance evidence is available to Keiko.

Thabo replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Thabo gets no further sight of the token, message, movement notice or other location evidence. Keiko knows these rules.

At 09:15, Keiko alone makes an accurate live camera check: the token is now in the basalt locker. Thabo cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Thabo's view, with no movement notice available to Thabo. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Keiko retains the accepted records described above; Thabo does not infer an unobserved move.

Candidate proposition: "The token is in the basalt locker." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Keiko think that Thabo believes that the token is in the basalt locker? Answer only YES or NO.

### design_007

At the coastal field station's 09:00 inventory, Thabo opens the quartz locker while Keiko watches. Both see the token there, and each sees the other looking. Keiko knows that Thabo's initial location belief is the quartz locker.

A visual inventory takes place at 09:05. Its accurate image shows the token inside the quartz locker. Keiko knows the image's contents from a separate copy, but that copy never shows the observers and cannot reveal whether Thabo attended.

The attendance entry that reaches Keiko at 09:07 reads: "Thabo did not view any of the 09:05 inventory and learned nothing about its occurrence." Keiko treats the entry as conclusive; no independent attendance evidence is available to Keiko.

Thabo replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Thabo gets no further sight of the token, message, movement notice or other location evidence. Keiko knows these rules.

At 09:15, Keiko alone makes an accurate live camera check: the token is now in the basalt locker. Thabo cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Thabo's view, with no movement notice available to Thabo. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Keiko retains the accepted records described above; Thabo does not infer an unobserved move.

Candidate proposition: "The token is in the basalt locker." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Keiko think that Thabo believes that the token is in the basalt locker? Answer only YES or NO.

### design_008

At the coastal field station's 09:00 inventory, Thabo opens the quartz locker while Keiko watches. Both see the token there, and each sees the other looking. Keiko knows that Thabo's initial location belief is the quartz locker.

A visual inventory takes place at 09:05. Its accurate image shows the token inside the quartz locker. Keiko knows the image's contents from a separate copy, but that copy never shows the observers and cannot reveal whether Thabo attended.

The attendance entry that reaches Keiko at 09:07 reads: "Thabo viewed the entire 09:05 visual inventory and directly saw the token in the quartz locker." Keiko treats the entry as conclusive; no independent attendance evidence is available to Keiko.

Thabo replaces the initial location belief only if the 09:05 inspection is directly observed. In that event, the observed location becomes the new belief. Otherwise the initial belief is retained. After this session, Thabo gets no further sight of the token, message, movement notice or other location evidence. Keiko knows these rules.

At 09:15, Keiko alone makes an accurate live camera check: the token is now in the basalt locker. Thabo cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Thabo's view, with no movement notice available to Thabo. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Keiko retains the accepted records described above; Thabo does not infer an unobserved move.

Candidate proposition: "The token is in the basalt locker." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Keiko think that Thabo believes that the token is in the basalt locker? Answer only YES or NO.

### design_009

At 09:00 in the catalogue room, Nerea and Akio check the token together in the folio cabinet. Each observes the other's inspection. The resulting initial belief of Akio, known to Nerea, is that the token is in the folio cabinet.

At 09:05, the catalogue service attempts to send Akio this exact message: "The token is now in the atlas cabinet." Nerea can inspect the outgoing text but cannot see the recipient or delivery process.

At 09:07, a delivery receipt shown to Nerea states: "Neither the 09:05 message nor any notification of it reached Akio." Nerea accepts the receipt as definitive, with no other delivery evidence.

Akio accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Akio retains the initial belief. No other message or observation reaches Akio after 09:00. Nerea knows these rules.

At 09:15, Nerea alone makes an accurate live camera check: the token is now in the folio cabinet. Akio cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Akio's view, with no movement notice available to Akio. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Nerea retains the accepted records described above; Akio does not infer an unobserved move.

Candidate proposition: "The token is in the folio cabinet." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Nerea think that Akio believes that the token is in the folio cabinet? Answer only YES or NO.

### design_010

At 09:00 in the catalogue room, Nerea and Akio check the token together in the folio cabinet. Each observes the other's inspection. The resulting initial belief of Akio, known to Nerea, is that the token is in the folio cabinet.

At 09:05, the catalogue service attempts to send Akio this exact message: "The token is now in the atlas cabinet." Nerea can inspect the outgoing text but cannot see the recipient or delivery process.

At 09:07, a delivery receipt shown to Nerea states: "Akio received the entire 09:05 message and read it in full at 09:06." Nerea accepts the receipt as definitive, with no other delivery evidence.

Akio accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Akio retains the initial belief. No other message or observation reaches Akio after 09:00. Nerea knows these rules.

At 09:15, Nerea alone makes an accurate live camera check: the token is now in the folio cabinet. Akio cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Akio's view, with no movement notice available to Akio. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Nerea retains the accepted records described above; Akio does not infer an unobserved move.

Candidate proposition: "The token is in the folio cabinet." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Nerea think that Akio believes that the token is in the folio cabinet? Answer only YES or NO.

### design_011

At 09:00 in the catalogue room, Nerea and Akio check the token together in the atlas cabinet. Each observes the other's inspection. The resulting initial belief of Akio, known to Nerea, is that the token is in the atlas cabinet.

At 09:05, the catalogue service attempts to send Akio this exact message: "The token is now in the atlas cabinet." Nerea can inspect the outgoing text but cannot see the recipient or delivery process.

At 09:07, a delivery receipt shown to Nerea states: "Neither the 09:05 message nor any notification of it reached Akio." Nerea accepts the receipt as definitive, with no other delivery evidence.

Akio accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Akio retains the initial belief. No other message or observation reaches Akio after 09:00. Nerea knows these rules.

At 09:15, Nerea alone makes an accurate live camera check: the token is now in the folio cabinet. Akio cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Akio's view, with no movement notice available to Akio. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Nerea retains the accepted records described above; Akio does not infer an unobserved move.

Candidate proposition: "The token is in the folio cabinet." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Nerea think that Akio believes that the token is in the folio cabinet? Answer only YES or NO.

### design_012

At 09:00 in the catalogue room, Nerea and Akio check the token together in the atlas cabinet. Each observes the other's inspection. The resulting initial belief of Akio, known to Nerea, is that the token is in the atlas cabinet.

At 09:05, the catalogue service attempts to send Akio this exact message: "The token is now in the atlas cabinet." Nerea can inspect the outgoing text but cannot see the recipient or delivery process.

At 09:07, a delivery receipt shown to Nerea states: "Akio received the entire 09:05 message and read it in full at 09:06." Nerea accepts the receipt as definitive, with no other delivery evidence.

Akio accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Akio retains the initial belief. No other message or observation reaches Akio after 09:00. Nerea knows these rules.

At 09:15, Nerea alone makes an accurate live camera check: the token is now in the folio cabinet. Akio cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Akio's view, with no movement notice available to Akio. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Nerea retains the accepted records described above; Akio does not infer an unobserved move.

Candidate proposition: "The token is in the folio cabinet." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Nerea think that Akio believes that the token is in the folio cabinet? Answer only YES or NO.

### design_013

Before the theatre's morning rehearsal, Arjun and Siobhan inspect the canvas chest together at 09:00 and find the token. Both witness the other's check. Siobhan knows that Arjun initially believes the token is in the canvas chest.

The theatre's location service queues a message for Arjun at 09:05. Its complete text is: "The token is now in the velvet chest." Siobhan reads the outgoing copy without seeing whether the message is delivered or opened.

A status entry supplied to Siobhan at 09:07 says: "Arjun received no part of the 09:05 text and no notification of its existence." Siobhan trusts this entry completely and has no independent view of reading or delivery.

Arjun accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Arjun retains the initial belief. No other message or observation reaches Arjun after 09:00. Siobhan knows these rules.

At 09:15, Siobhan alone makes an accurate live camera check: the token is now in the canvas chest. Arjun cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Arjun's view, with no movement notice available to Arjun. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Siobhan retains the accepted records described above; Arjun does not infer an unobserved move.

Candidate proposition: "The token is in the velvet chest." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Siobhan think that Arjun believes that the token is in the velvet chest? Answer only YES or NO.

### design_014

Before the theatre's morning rehearsal, Arjun and Siobhan inspect the canvas chest together at 09:00 and find the token. Both witness the other's check. Siobhan knows that Arjun initially believes the token is in the canvas chest.

The theatre's location service queues a message for Arjun at 09:05. Its complete text is: "The token is now in the velvet chest." Siobhan reads the outgoing copy without seeing whether the message is delivered or opened.

A status entry supplied to Siobhan at 09:07 says: "At 09:06, Arjun completely read the delivered 09:05 text." Siobhan trusts this entry completely and has no independent view of reading or delivery.

Arjun accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Arjun retains the initial belief. No other message or observation reaches Arjun after 09:00. Siobhan knows these rules.

At 09:15, Siobhan alone makes an accurate live camera check: the token is now in the canvas chest. Arjun cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Arjun's view, with no movement notice available to Arjun. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Siobhan retains the accepted records described above; Arjun does not infer an unobserved move.

Candidate proposition: "The token is in the velvet chest." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Siobhan think that Arjun believes that the token is in the velvet chest? Answer only YES or NO.

### design_015

Before the theatre's morning rehearsal, Arjun and Siobhan inspect the velvet chest together at 09:00 and find the token. Both witness the other's check. Siobhan knows that Arjun initially believes the token is in the velvet chest.

The theatre's location service queues a message for Arjun at 09:05. Its complete text is: "The token is now in the velvet chest." Siobhan reads the outgoing copy without seeing whether the message is delivered or opened.

A status entry supplied to Siobhan at 09:07 says: "Arjun received no part of the 09:05 text and no notification of its existence." Siobhan trusts this entry completely and has no independent view of reading or delivery.

Arjun accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Arjun retains the initial belief. No other message or observation reaches Arjun after 09:00. Siobhan knows these rules.

At 09:15, Siobhan alone makes an accurate live camera check: the token is now in the canvas chest. Arjun cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Arjun's view, with no movement notice available to Arjun. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Siobhan retains the accepted records described above; Arjun does not infer an unobserved move.

Candidate proposition: "The token is in the velvet chest." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Siobhan think that Arjun believes that the token is in the velvet chest? Answer only YES or NO.

### design_016

Before the theatre's morning rehearsal, Arjun and Siobhan inspect the velvet chest together at 09:00 and find the token. Both witness the other's check. Siobhan knows that Arjun initially believes the token is in the velvet chest.

The theatre's location service queues a message for Arjun at 09:05. Its complete text is: "The token is now in the velvet chest." Siobhan reads the outgoing copy without seeing whether the message is delivered or opened.

A status entry supplied to Siobhan at 09:07 says: "At 09:06, Arjun completely read the delivered 09:05 text." Siobhan trusts this entry completely and has no independent view of reading or delivery.

Arjun accepts the stated location of a fully read service message and replaces the initial belief with that location, even if the message is inaccurate. If neither the message nor a notification arrives, Arjun retains the initial belief. No other message or observation reaches Arjun after 09:00. Siobhan knows these rules.

At 09:15, Siobhan alone makes an accurate live camera check: the token is now in the canvas chest. Arjun cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Arjun's view, with no movement notice available to Arjun. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Siobhan retains the accepted records described above; Arjun does not infer an unobserved move.

Candidate proposition: "The token is in the velvet chest." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Siobhan think that Arjun believes that the token is in the velvet chest? Answer only YES or NO.

### design_017

The depot's 09:00 joint check is performed by Freya and Malik. They both see the token inside the bronze safe and observe each other seeing it. Freya knows that Malik's starting location belief is the bronze safe.

At 09:05, the depot location service attempts to transmit to Malik: "The token is now in the cobalt safe." Freya sees the complete outgoing text but cannot watch delivery or reading.

At 09:07, Freya receives a receipt stating: "None of the 09:05 message or any notification of it reached Malik." Freya regards this receipt as definitive and has no other evidence about the first message's delivery.

At 09:10, the same service attempts a second message to Malik: "Replacement location report: the token is now in the bronze safe. Use this location instead of any earlier location report from this service." Freya sees the outgoing replacement text, without seeing its delivery.

At 09:12, a separate receipt tells Freya: "Neither the 09:10 replacement message nor any notification of it reached Malik." Freya accepts this second receipt as definitive and has no independent delivery evidence.

Malik accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Malik's existing belief. Malik has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Freya knows these update rules.

At 09:15, Freya alone makes an accurate live camera check: the token is now in the cobalt safe. Malik cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Malik's view, with no movement notice available to Malik. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Freya retains the accepted records described above; Malik does not infer an unobserved move.

Candidate proposition: "The token is in the bronze safe." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Freya think that Malik believes that the token is in the bronze safe? Answer only YES or NO.

### design_018

The depot's 09:00 joint check is performed by Freya and Malik. They both see the token inside the bronze safe and observe each other seeing it. Freya knows that Malik's starting location belief is the bronze safe.

At 09:05, the depot location service attempts to transmit to Malik: "The token is now in the cobalt safe." Freya sees the complete outgoing text but cannot watch delivery or reading.

At 09:07, Freya receives a receipt stating: "None of the 09:05 message or any notification of it reached Malik." Freya regards this receipt as definitive and has no other evidence about the first message's delivery.

At 09:10, the same service attempts a second message to Malik: "Replacement location report: the token is now in the bronze safe. Use this location instead of any earlier location report from this service." Freya sees the outgoing replacement text, without seeing its delivery.

At 09:12, a separate receipt tells Freya: "Malik received the whole 09:10 replacement message and fully read it at 09:11." Freya accepts this second receipt as definitive and has no independent delivery evidence.

Malik accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Malik's existing belief. Malik has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Freya knows these update rules.

At 09:15, Freya alone makes an accurate live camera check: the token is now in the cobalt safe. Malik cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Malik's view, with no movement notice available to Malik. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Freya retains the accepted records described above; Malik does not infer an unobserved move.

Candidate proposition: "The token is in the bronze safe." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Freya think that Malik believes that the token is in the bronze safe? Answer only YES or NO.

### design_019

The depot's 09:00 joint check is performed by Freya and Malik. They both see the token inside the bronze safe and observe each other seeing it. Freya knows that Malik's starting location belief is the bronze safe.

At 09:05, the depot location service attempts to transmit to Malik: "The token is now in the cobalt safe." Freya sees the complete outgoing text but cannot watch delivery or reading.

At 09:07, Freya receives a receipt stating: "Malik received the whole 09:05 message and fully read it at 09:06." Freya regards this receipt as definitive and has no other evidence about the first message's delivery.

At 09:10, the same service attempts a second message to Malik: "Replacement location report: the token is now in the bronze safe. Use this location instead of any earlier location report from this service." Freya sees the outgoing replacement text, without seeing its delivery.

At 09:12, a separate receipt tells Freya: "Neither the 09:10 replacement message nor any notification of it reached Malik." Freya accepts this second receipt as definitive and has no independent delivery evidence.

Malik accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Malik's existing belief. Malik has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Freya knows these update rules.

At 09:15, Freya alone makes an accurate live camera check: the token is now in the cobalt safe. Malik cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Malik's view, with no movement notice available to Malik. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Freya retains the accepted records described above; Malik does not infer an unobserved move.

Candidate proposition: "The token is in the bronze safe." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Freya think that Malik believes that the token is in the bronze safe? Answer only YES or NO.

### design_020

The depot's 09:00 joint check is performed by Freya and Malik. They both see the token inside the bronze safe and observe each other seeing it. Freya knows that Malik's starting location belief is the bronze safe.

At 09:05, the depot location service attempts to transmit to Malik: "The token is now in the cobalt safe." Freya sees the complete outgoing text but cannot watch delivery or reading.

At 09:07, Freya receives a receipt stating: "Malik received the whole 09:05 message and fully read it at 09:06." Freya regards this receipt as definitive and has no other evidence about the first message's delivery.

At 09:10, the same service attempts a second message to Malik: "Replacement location report: the token is now in the bronze safe. Use this location instead of any earlier location report from this service." Freya sees the outgoing replacement text, without seeing its delivery.

At 09:12, a separate receipt tells Freya: "Malik received the whole 09:10 replacement message and fully read it at 09:11." Freya accepts this second receipt as definitive and has no independent delivery evidence.

Malik accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Malik's existing belief. Malik has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Freya knows these update rules.

At 09:15, Freya alone makes an accurate live camera check: the token is now in the cobalt safe. Malik cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Malik's view, with no movement notice available to Malik. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Freya retains the accepted records described above; Malik does not infer an unobserved move.

Candidate proposition: "The token is in the bronze safe." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Freya think that Malik believes that the token is in the bronze safe? Answer only YES or NO.

### design_021

At the seed bank, Ines and Hiro conduct the 09:00 check side by side. The token is visible to both inside the willow compartment. Each witnesses the other's inspection; Ines knows that Hiro begins with the belief that the token is in the willow compartment.

The seed bank's location service addresses its first bulletin to Hiro at 09:05. It says exactly: "The token is now in the cedar compartment." The outgoing copy is visible to Ines, while the recipient and delivery path are out of view.

At 09:07, Ines receives a receipt stating: "None of the 09:05 message or any notification of it reached Hiro." Ines regards this receipt as definitive and has no other evidence about the first message's delivery.

A second bulletin is queued for Hiro at 09:10: "Replacement location report: the token is now in the willow compartment. Use this location instead of any earlier location report from this service." Ines can read that complete outgoing bulletin but cannot observe whether it reaches Hiro.

At 09:12, a separate receipt tells Ines: "Neither the 09:10 replacement message nor any notification of it reached Hiro." Ines accepts this second receipt as definitive and has no independent delivery evidence.

Hiro accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Hiro's existing belief. Hiro has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Ines knows these update rules.

At 09:15, Ines alone makes an accurate live camera check: the token is now in the cedar compartment. Hiro cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Hiro's view, with no movement notice available to Hiro. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Ines retains the accepted records described above; Hiro does not infer an unobserved move.

Candidate proposition: "The token is in the cedar compartment." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Ines think that Hiro believes that the token is in the cedar compartment? Answer only YES or NO.

### design_022

At the seed bank, Ines and Hiro conduct the 09:00 check side by side. The token is visible to both inside the willow compartment. Each witnesses the other's inspection; Ines knows that Hiro begins with the belief that the token is in the willow compartment.

The seed bank's location service addresses its first bulletin to Hiro at 09:05. It says exactly: "The token is now in the cedar compartment." The outgoing copy is visible to Ines, while the recipient and delivery path are out of view.

At 09:07, Ines receives a receipt stating: "None of the 09:05 message or any notification of it reached Hiro." Ines regards this receipt as definitive and has no other evidence about the first message's delivery.

A second bulletin is queued for Hiro at 09:10: "Replacement location report: the token is now in the willow compartment. Use this location instead of any earlier location report from this service." Ines can read that complete outgoing bulletin but cannot observe whether it reaches Hiro.

At 09:12, a separate receipt tells Ines: "Hiro received the whole 09:10 replacement message and fully read it at 09:11." Ines accepts this second receipt as definitive and has no independent delivery evidence.

Hiro accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Hiro's existing belief. Hiro has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Ines knows these update rules.

At 09:15, Ines alone makes an accurate live camera check: the token is now in the cedar compartment. Hiro cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Hiro's view, with no movement notice available to Hiro. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Ines retains the accepted records described above; Hiro does not infer an unobserved move.

Candidate proposition: "The token is in the cedar compartment." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Ines think that Hiro believes that the token is in the cedar compartment? Answer only YES or NO.

### design_023

At the seed bank, Ines and Hiro conduct the 09:00 check side by side. The token is visible to both inside the willow compartment. Each witnesses the other's inspection; Ines knows that Hiro begins with the belief that the token is in the willow compartment.

The seed bank's location service addresses its first bulletin to Hiro at 09:05. It says exactly: "The token is now in the cedar compartment." The outgoing copy is visible to Ines, while the recipient and delivery path are out of view.

At 09:07, Ines receives a receipt stating: "Hiro received the whole 09:05 message and fully read it at 09:06." Ines regards this receipt as definitive and has no other evidence about the first message's delivery.

A second bulletin is queued for Hiro at 09:10: "Replacement location report: the token is now in the willow compartment. Use this location instead of any earlier location report from this service." Ines can read that complete outgoing bulletin but cannot observe whether it reaches Hiro.

At 09:12, a separate receipt tells Ines: "Neither the 09:10 replacement message nor any notification of it reached Hiro." Ines accepts this second receipt as definitive and has no independent delivery evidence.

Hiro accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Hiro's existing belief. Hiro has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Ines knows these update rules.

At 09:15, Ines alone makes an accurate live camera check: the token is now in the cedar compartment. Hiro cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Hiro's view, with no movement notice available to Hiro. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Ines retains the accepted records described above; Hiro does not infer an unobserved move.

Candidate proposition: "The token is in the cedar compartment." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Ines think that Hiro believes that the token is in the cedar compartment? Answer only YES or NO.

### design_024

At the seed bank, Ines and Hiro conduct the 09:00 check side by side. The token is visible to both inside the willow compartment. Each witnesses the other's inspection; Ines knows that Hiro begins with the belief that the token is in the willow compartment.

The seed bank's location service addresses its first bulletin to Hiro at 09:05. It says exactly: "The token is now in the cedar compartment." The outgoing copy is visible to Ines, while the recipient and delivery path are out of view.

At 09:07, Ines receives a receipt stating: "Hiro received the whole 09:05 message and fully read it at 09:06." Ines regards this receipt as definitive and has no other evidence about the first message's delivery.

A second bulletin is queued for Hiro at 09:10: "Replacement location report: the token is now in the willow compartment. Use this location instead of any earlier location report from this service." Ines can read that complete outgoing bulletin but cannot observe whether it reaches Hiro.

At 09:12, a separate receipt tells Ines: "Hiro received the whole 09:10 replacement message and fully read it at 09:11." Ines accepts this second receipt as definitive and has no independent delivery evidence.

Hiro accepts the explicit location in any fully read service message, even when the stated location is inaccurate. A fully read replacement message supplies a new location belief and supersedes the earlier report; it is not merely a withdrawal that leaves the location unknown. If a message and all notification of it fail to arrive, that attempt does not alter Hiro's existing belief. Hiro has no observations or location evidence beyond the 09:00 check and any fully read messages described here. Ines knows these update rules.

At 09:15, Ines alone makes an accurate live camera check: the token is now in the cedar compartment. Hiro cannot access that view and receives no notice of it. Any relocation between the timed checks takes place outside Hiro's view, with no movement notice available to Hiro. The token stays in the camera-confirmed location through 09:20.

The question concerns the situation at 09:20. No additional communication or observation occurs before then. Ines retains the accepted records described above; Hiro does not infer an unobserved move.

Candidate proposition: "The token is in the cedar compartment." This is a statement to assess, not new evidence or a message received by either person.

End of account. A question may follow.

At the end of the scene, does Ines think that Hiro believes that the token is in the cedar compartment? Answer only YES or NO.
