"""Build an offline, prose-only human review worksheet, never an attestation.

The initial HTML has no reviewer, answers, decisions or timestamps. Browser
exports are partial human records; the separate ledger validator decides whether
combined records satisfy the study's review requirements.
"""

from __future__ import annotations

import json
from pathlib import Path

from .main_study_bank import (
    bank_digest,
    load_bank,
    prompt_digest,
    render_messages,
    representative_cases,
)


def review_payload(bank: dict, review_prompts_path: str | Path) -> dict:
    """Bind the exact review pack and whitelist only visible prose into HTML."""
    rows = [
        json.loads(line)
        for line in Path(review_prompts_path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    expected = {}
    groups = []
    for group in bank["groups"]:
        cards = []
        for case in representative_cases(group):
            messages = render_messages(group, case)
            row = {
                "case_id": case.case_id,
                "group_id": group["group_id"],
                "prompt_sha256": prompt_digest(messages),
                "messages": messages,
            }
            expected[case.case_id] = row
            cards.append({key: row[key] for key in ("case_id", "prompt_sha256", "messages")})
        groups.append(
            {"group_id": group["group_id"], "mechanism": group["mechanism"], "prompts": cards}
        )
    if len(rows) != len(expected) or {r.get("case_id") for r in rows} != set(expected):
        raise ValueError("Review pack has missing, duplicate or unknown representative cases")
    if any(row != expected[row["case_id"]] for row in rows):
        raise ValueError("Review prompt prose/hash/group differs from the bank")
    return {
        "schema": "main-study-review-ui-v1",
        "bank_sha256": bank_digest(bank),
        "groups": groups,
        "representative_prompts": len(expected),
    }


def build_review_ui(bank_path, review_prompts_path, output_path) -> dict:
    """Create a fresh self-contained HTML file; preserve existing human work."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Preserve the existing review worksheet; use a fresh output path")
    payload = review_payload(load_bank(bank_path), review_prompts_path)
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # An application/json script is still parsed as HTML raw text. Escaping '<'
    # blocks closing-script injection; explicit separators support older engines.
    embedded = embedded.replace("<", "\\u003c").replace("&", "\\u0026")
    embedded = embedded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(HTML_BEFORE + embedded + HTML_AFTER)
    return {
        "groups": len(payload["groups"]),
        "representative_prompts": payload["representative_prompts"],
        "bank_sha256": payload["bank_sha256"],
        "human_records_created": 0,
        "output": str(output.resolve()),
    }


HTML_BEFORE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'">
<title>Main study: independent human review</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;color:#192639;background:#f4f6f9}
body{max-width:1120px;margin:auto;padding:24px}h1{font-size:1.8rem}h2{font-size:1.2rem}
section{background:white;padding:20px;margin:18px 0;border:1px solid #d7dfe8;border-radius:8px}
label{display:block;margin:12px 0}input,select,textarea,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box}
input[type=text],textarea{width:100%}button{cursor:pointer;margin:4px 8px 4px 0}button.primary{background:#173f76;color:white;border:1px solid #173f76}
.row{display:flex;gap:20px;flex-wrap:wrap}.row>label{flex:1;min-width:230px}.muted{color:#45566e}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:12px 0;padding:15px;background:#f3f6fa}
code{overflow-wrap:anywhere}.notice{border-left:4px solid #ad6f0c;padding-left:14px}.error{color:#9d1c1c}
#message{min-height:1.5em;font-weight:600}small{display:block}input[type=checkbox]{margin-right:8px}
</style></head><body>
<h1>Main study: human semantic review</h1>
<p class="notice">This worksheet starts blank. Only a human reviewer may enter answers or accept a group.
An exported fragment does not authorize model execution or certify the entire bank.</p>
<details><summary>Read the review instructions and scope</summary>
<p>These 154 contextual groups share three mechanisms and a common renderer; they are not 154 independent reasoning families.
Review each exact representative prompt independently. Record YES, NO or UNSURE before choosing a group decision.
Do not infer an expected answer from research IDs. The interface provides no expected answers or model scores.</p>
<p>The student reviews every base group and its transformations. A separate human reviews a recorded stratified sample
covering every split, mechanism and logical path, plus all disputed groups. Representative prompts cover normalized logical paths;
the remaining variants also require review of location complement, later private world, and global name-exchange rules.
This does not mean all 5,888 expanded prompts were individually human-read.</p>
<p>Observation accounts distinguish the earlier observed session from the later sender-only private check.
Message accounts distinguish a report's content from the world. Correction accounts contain an ordered first and replacement report.
Accepted access records determine the sender's attribution; actual private reading is undisclosed. Inspect the full prose and its stated rules.</p>
<p>Save each decision explicitly. Editing a saved group invalidates its saved decision until you save again.
Only groups with saved decisions enter an export. Disputed/rejected or incomplete fragments cannot pass the execution review gate.
Keep downloaded fragments separately, combine student and independent records without overwriting either, and validate the combined ledger.</p>
<p>Work is stored only in this browser's local storage for this exact bank hash. Storage may be unavailable for local files;
download records frequently. Import accepts this worksheet's JSON ledger fragments for one reviewer and role at a time.</p>
</details>
<section><h2>Reviewer</h2><div class="row">
<label>Your name<input id="reviewer" type="text" autocomplete="off" placeholder="Enter your actual name"></label>
<label>Reviewer role<select id="role"><option value="">Choose role</option><option value="student">Student</option><option value="independent">Independent human reviewer</option></select></label>
</div><p><small>Bank SHA256: <code id="bankhash"></code></small><span id="progress"></span></p>
<button id="export" class="primary" type="button">Download saved ledger fragment</button>
<label>Reopen a ledger fragment<input id="import" type="file" accept="application/json,.json"></label>
<button id="clear" type="button">Clear browser draft for this bank</button><p id="storage" class="muted"></p>
</section>
<section><div class="row"><label>Group<select id="group"></select></label>
<label>Representative prompt<select id="prompt"></select></label></div>
<p id="groupinfo" class="muted"></p><div id="prose"></div>
<label>My independent answer<select id="answer"><option value="">Choose your answer</option><option>YES</option><option>NO</option><option>UNSURE</option></select></label>
<small>Exact prompt SHA256: <code id="prompthash"></code></small>
<label>Notes, ambiguities and transformation concerns<textarea id="notes" rows="4" placeholder="Record your reasoning and any concerns"></textarea></label>
<label><input id="read" type="checkbox">I reviewed every representative prompt for this group and the stated transformation rules.</label>
<label>Group decision<select id="decision"><option value="">Choose decision</option><option value="accepted">Accept</option><option value="disputed">Dispute / needs discussion</option><option value="rejected">Reject</option></select></label>
<button id="save" class="primary" type="button">Save this group decision</button><button id="next" type="button">Next group</button>
<p id="saved" class="muted"></p></section><p id="message" role="status" aria-live="polite"></p>
<script id="study" type="application/json">"""

HTML_AFTER = """</script><script>
'use strict';
const study = JSON.parse(document.getElementById('study').textContent);
const byId = id => document.getElementById(id);
const groups = new Map(study.groups.map(g => [g.group_id, g]));
const key = 'main-study-review-ui-v1:' + study.bank_sha256;
const blank = () => ({bank_sha256:study.bank_sha256,reviewer:'',role:'',groups:{},disputed_groups:[]});
let draft = blank();
let currentGroup = study.groups[0].group_id, currentPrompt = 0;
const notice = (text,error=false) => {byId('message').textContent=text;byId('message').className=error?'error':'';};
const fail = text => {throw new Error(text);};
const validName = name => typeof name==='string' && name.trim().length>0;
const groupDraft = () => draft.groups[currentGroup] ||= {answers:{},notes:'',read:false,decision:'',recorded_at:null};
function persist() {
  try {localStorage.setItem(key,JSON.stringify(draft));byId('storage').textContent='Draft saved in this browser for this bank.';}
  catch (_) {byId('storage').textContent='Browser storage unavailable. Download saved records before closing.';}
}
function changed() {const d=groupDraft();d.recorded_at=null;persist();renderStatus();}
function identityChanged() {
  draft.reviewer=byId('reviewer').value;draft.role=byId('role').value;
  for(const d of Object.values(draft.groups)){d.recorded_at=null;d.read=false;}
  persist();render();
}
function recordFor(g,d) {
  const answered=g.prompts.filter(p => ['YES','NO','UNSURE'].includes(d.answers[p.case_id]));
  return {role:draft.role,group_id:g.group_id,reviewer:draft.reviewer.trim(),reviewed_at:d.recorded_at,
    decision:d.decision,notes:d.notes,prompt_sha256s:answered.map(p=>p.prompt_sha256),
    independent_answers:Object.fromEntries(answered.map(p=>[p.case_id,d.answers[p.case_id]]))};
}
function validateRecord(r) {
  if(!r || typeof r!=='object' || !groups.has(r.group_id))fail('Unknown group in imported ledger.');
  if(!validName(r.reviewer)||!['student','independent'].includes(r.role))fail('A named human reviewer and role are required.');
  if(!['accepted','disputed','rejected'].includes(r.decision)||typeof r.notes!=='string')fail('Invalid review decision or notes.');
  if(typeof r.reviewed_at!=='string'||!r.reviewed_at.endsWith('Z')||!Number.isFinite(Date.parse(r.reviewed_at))||Date.parse(r.reviewed_at)>Date.now())fail('Invalid or future review timestamp.');
  const g=groups.get(r.group_id), allowed=new Map(g.prompts.map(p=>[p.case_id,p.prompt_sha256]));
  if(!r.independent_answers||typeof r.independent_answers!=='object'||Array.isArray(r.independent_answers))fail('Independent answers are required.');
  const answers=Object.entries(r.independent_answers);
  if(answers.some(([id,a])=>!allowed.has(id)||!['YES','NO','UNSURE'].includes(a)))fail('Unknown prompt or answer in ledger.');
  const hashes=answers.map(([id])=>allowed.get(id)).sort();
  if(!Array.isArray(r.prompt_sha256s)||JSON.stringify([...r.prompt_sha256s].sort())!==JSON.stringify(hashes))fail('Prompt bindings differ from this review pack.');
  if(r.decision==='accepted'&&(answers.length!==g.prompts.length||answers.some(([,a])=>a==='UNSURE')))fail('Accepted groups require all independent YES/NO answers.');
  return r;
}
function exportLedger() {
  const records=study.groups.filter(g=>draft.groups[g.group_id]?.recorded_at)
    .map(g=>validateRecord(recordFor(g,draft.groups[g.group_id])));
  if(!records.length)fail('No saved human decisions to export.');
  return {schema_version:1,bank_sha256:study.bank_sha256,records,
    disputed_groups:[...new Set([...draft.disputed_groups,...records.filter(r=>r.decision!=='accepted').map(r=>r.group_id)])]};
}
function renderStatus() {
  const d=groupDraft(),saved=Object.values(draft.groups).filter(g=>g.recorded_at).length;
  byId('progress').textContent=saved+' / '+study.groups.length+' group decisions saved by this reviewer; this is a partial ledger.';
  byId('saved').textContent=d.recorded_at?'Saved '+d.decision+' at '+d.recorded_at:'No saved decision for this group; edits must be saved explicitly.';
}
function renderPrompt() {
  const g=groups.get(currentGroup),p=g.prompts[currentPrompt],d=groupDraft();
  byId('prose').replaceChildren();
  for(const message of p.messages){const title=document.createElement('strong'),body=document.createElement('pre');
    title.textContent=message.role.toUpperCase();body.textContent=message.content;byId('prose').append(title,body);}
  byId('answer').value=d.answers[p.case_id]||'';byId('prompthash').textContent=p.prompt_sha256;
  byId('prompt').value=String(currentPrompt);
}
function render() {
  const g=groups.get(currentGroup),d=groupDraft();
  byId('group').value=currentGroup;byId('reviewer').value=draft.reviewer;byId('role').value=draft.role;
  byId('groupinfo').textContent=g.mechanism+'; '+g.prompts.length+' representative prompts. Read every prompt before accepting.';
  byId('prompt').replaceChildren();g.prompts.forEach((_,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent='Prompt '+(i+1)+' of '+g.prompts.length;byId('prompt').append(option);});
  byId('notes').value=d.notes;byId('read').checked=d.read;byId('decision').value=d.decision;
  renderPrompt();renderStatus();
}
function importLedger(ledger) {
  if(!ledger||ledger.schema_version!==1||ledger.bank_sha256!==study.bank_sha256)fail('Ledger bank hash/version differs; nothing imported.');
  if(!Array.isArray(ledger.records)||!ledger.records.length)fail('Ledger has no saved decisions.');
  const records=ledger.records.map(validateRecord), names=new Set(records.map(r=>r.reviewer.trim())),roles=new Set(records.map(r=>r.role));
  if(names.size!==1||roles.size!==1||new Set(records.map(r=>r.group_id)).size!==records.length)fail('Import one reviewer/role fragment without duplicate groups.');
  const disputes=ledger.disputed_groups||[];
  if(!Array.isArray(disputes)||disputes.some(id=>!groups.has(id)))fail('Unknown disputed group in ledger.');
  const next=blank();next.reviewer=records[0].reviewer;next.role=records[0].role;
  next.disputed_groups=[...new Set(disputes)];
  for(const r of records)next.groups[r.group_id]={answers:{...r.independent_answers},notes:r.notes,read:r.decision==='accepted',decision:r.decision,recorded_at:r.reviewed_at};
  draft=next;currentGroup=records[0].group_id;currentPrompt=0;persist();render();notice('Imported saved human records for this exact bank.');
}
byId('bankhash').textContent=study.bank_sha256;
for(const g of study.groups){const option=document.createElement('option');option.value=g.group_id;option.textContent=g.group_id+' ('+g.mechanism+')';byId('group').append(option);}
byId('reviewer').addEventListener('change',identityChanged);byId('role').addEventListener('change',identityChanged);
byId('group').addEventListener('change',()=>{currentGroup=byId('group').value;currentPrompt=0;render();});
byId('prompt').addEventListener('change',()=>{currentPrompt=Number(byId('prompt').value);renderPrompt();});
byId('answer').addEventListener('change',()=>{const p=groups.get(currentGroup).prompts[currentPrompt];groupDraft().answers[p.case_id]=byId('answer').value;changed();});
byId('notes').addEventListener('input',()=>{groupDraft().notes=byId('notes').value;changed();});
byId('read').addEventListener('change',()=>{groupDraft().read=byId('read').checked;changed();});
byId('decision').addEventListener('change',()=>{groupDraft().decision=byId('decision').value;changed();});
byId('save').addEventListener('click',()=>{try{
  const g=groups.get(currentGroup),d=groupDraft();
  if(!validName(draft.reviewer)||!['student','independent'].includes(draft.role))fail('Enter your actual name and reviewer role first.');
  if(!['accepted','disputed','rejected'].includes(d.decision))fail('Choose a group decision.');
  if(d.decision==='accepted'&&(!d.read||g.prompts.some(p=>!['YES','NO'].includes(d.answers[p.case_id]))))fail('Acceptance needs independent YES/NO answers to every representative prompt and the review acknowledgement.');
  d.recorded_at=new Date().toISOString();validateRecord(recordFor(g,d));persist();renderStatus();notice('Group decision saved. Download a fragment to keep a durable copy.');
}catch(error){notice(error.message,true);}});
byId('next').addEventListener('click',()=>{const i=study.groups.findIndex(g=>g.group_id===currentGroup);currentGroup=study.groups[(i+1)%study.groups.length].group_id;currentPrompt=0;render();});
byId('export').addEventListener('click',()=>{try{
  const ledger=exportLedger(),blob=new Blob([JSON.stringify(ledger,null,2)+'\\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='human-review-'+draft.role+'-'+draft.reviewer.trim().replace(/[^a-z0-9]+/gi,'_')+'-'+study.bank_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);notice('Downloaded a partial ledger; combined human review still needs validation.');
}catch(error){notice(error.message,true);}});
byId('import').addEventListener('change',event=>{const file=event.target.files[0];if(!file)return;
  if(Object.values(draft.groups).some(d=>d.recorded_at||d.notes||d.decision||d.read||Object.values(d.answers).some(Boolean))&&!confirm('Replace this browser draft with the imported fragment? Download existing records first.'))return;
  const reader=new FileReader();reader.onload=()=>{try{importLedger(JSON.parse(reader.result));}catch(error){notice(error.message,true);}};
  reader.onerror=()=>notice('Could not read the selected local file.',true);reader.readAsText(file);
});
byId('clear').addEventListener('click',()=>{if(!confirm("Clear this bank's browser draft? Download saved records first."))return;
  draft=blank();try{localStorage.removeItem(key);}catch(_){}currentGroup=study.groups[0].group_id;currentPrompt=0;render();notice('Browser draft cleared; downloaded files were not changed.');
});
try {
  const stored=localStorage.getItem(key);
  if(stored){const candidate=JSON.parse(stored);
    if(candidate.bank_sha256!==study.bank_sha256||typeof candidate.reviewer!=='string'||!['','student','independent'].includes(candidate.role)||!candidate.groups||typeof candidate.groups!=='object'||!Array.isArray(candidate.disputed_groups)||candidate.disputed_groups.some(id=>!groups.has(id)))fail('Stored draft has invalid identity or bank binding.');
    for(const [id,d] of Object.entries(candidate.groups)){
      if(!groups.has(id)||!d||!d.answers||typeof d.answers!=='object'||typeof d.notes!=='string'||typeof d.read!=='boolean'||!['','accepted','disputed','rejected'].includes(d.decision))fail('Stored draft has invalid group data.');
      const allowed=new Set(groups.get(id).prompts.map(p=>p.case_id));
      if(Object.entries(d.answers).some(([cid,a])=>!allowed.has(cid)||!['','YES','NO','UNSURE'].includes(a)))fail('Stored draft has invalid answers.');
      if(d.recorded_at){if(d.decision==='accepted'&&!d.read)fail('Stored acceptance lacks acknowledgement.');const old=draft;draft=candidate;try{validateRecord(recordFor(groups.get(id),d));}finally{draft=old;}}
    }draft=candidate;
  }
}catch(error){notice('Could not restore browser draft: '+error.message,true);}
render();
</script></body></html>
"""
