"""Blinding, stale-input and offline review-export safeguards on CPU fixtures."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter

import pytest
from test_main_study_bank import synthetic_bank as synthetic_bank

from latent_probes.main_study_bank import bank_digest, export_review
from latent_probes.main_study_review_ui import HTML_AFTER, build_review_ui, review_payload


@pytest.fixture
def pack(synthetic_bank, tmp_path):
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(synthetic_bank), encoding="utf-8")
    review = tmp_path / "review"
    export_review(synthetic_bank, review)
    return synthetic_bank, bank_path, review / "review_prompts.jsonl", tmp_path / "review.html"


def embedded_payload(text):
    return json.loads(
        re.search(r'<script id="study" type="application/json">(.*?)</script>', text, re.S).group(1)
    )


def test_ui_has_exact_154_groups_and_736_bound_prose_prompts(pack):
    bank, bank_path, prompts, output = pack
    result = build_review_ui(bank_path, prompts, output)
    assert result["groups"] == 154 and result["representative_prompts"] == 736
    assert result["human_records_created"] == 0
    payload = embedded_payload(output.read_text(encoding="utf-8"))
    assert payload["bank_sha256"] == bank_digest(bank)
    assert len(payload["groups"]) == 154
    assert Counter(len(g["prompts"]) for g in payload["groups"]) == {4: 124, 8: 30}
    for group in payload["groups"]:
        assert set(group) == {"group_id", "mechanism", "prompts"}
        for row in group["prompts"]:
            assert set(row) == {"case_id", "prompt_sha256", "messages"}
            assert all(set(m) == {"role", "content"} for m in row["messages"])


def test_no_structured_answers_or_fake_humans_are_embedded(pack):
    _, bank_path, prompts, output = pack
    build_review_ui(bank_path, prompts, output)
    text = output.read_text(encoding="utf-8")
    payload = json.dumps(embedded_payload(text))
    for forbidden in (
        '"gold":',
        '"state":',
        '"prediction":',
        '"reviewer":',
        '"decision":',
        '"reviewed_at":',
        '"independent_answers":',
    ):
        assert forbidden not in payload
    assert "reviewer:'',role:''" in text
    assert "new Date().toISOString()" in text
    assert "Acceptance needs independent YES/NO answers" in text
    assert "a partial ledger" in text


@pytest.mark.parametrize("corruption", ("hash", "messages", "missing", "duplicate", "hidden"))
def test_stale_or_augmented_review_pack_is_rejected(pack, corruption):
    bank, _, prompts, _ = pack
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines()]
    if corruption == "hash":
        rows[0]["prompt_sha256"] = "0" * 64
    elif corruption == "messages":
        rows[0]["messages"][0]["content"] += " stale change"
    elif corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[0]["gold"] = 1
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Review"):
        review_payload(bank, prompts)


def test_existing_worksheet_is_not_overwritten(pack):
    _, bank_path, prompts, output = pack
    output.write_text("Existing human work", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_review_ui(bank_path, prompts, output)
    assert output.read_text(encoding="utf-8") == "Existing human work"


def test_offline_html_has_no_network_or_untrusted_html_sinks(pack):
    _, bank_path, prompts, output = pack
    build_review_ui(bank_path, prompts, output)
    text = output.read_text(encoding="utf-8")
    assert "connect-src 'none'" in text
    for forbidden in (
        "fetch(",
        "XMLHttpRequest",
        "innerHTML",
        "document.write",
        "<iframe",
        "<link",
        "https://",
        "http://",
        "eval(",
    ):
        assert forbidden not in text
    assert "body.textContent=message.content" in text
    assert "localStorage.setItem(key" in text and "catch (_)" in text
    assert "confirm(" in text and "FileReader" in text


def test_closing_script_and_unicode_separators_cannot_break_embedded_data(pack):
    bank, bank_path, prompts, output = pack
    bank["groups"][0]["introduction"] += ' </script><script>alert("source")</script> & \u2028\u2029'
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    # A regenerated pack is necessary: original changed prose is rejected.
    changed_pack = prompts.parent / "changed_pack"
    export_review(bank, changed_pack)
    prompts = changed_pack / "review_prompts.jsonl"
    build_review_ui(bank_path, prompts, output)
    text = output.read_text(encoding="utf-8")
    assert text.count("</script>") == 2
    assert "\\u003c/script>" in text and "\\u2028" in text and "\\u2029" in text
    restored = embedded_payload(text)
    assert any(
        '<script>alert("source")</script>' in message["content"]
        for group in restored["groups"]
        for prompt in group["prompts"]
        for message in prompt["messages"]
    )


def test_inline_javascript_parses_without_a_browser(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Optional Node syntax check; no Node runtime in this environment")
    path = tmp_path / "review.js"
    path.write_text(HTML_AFTER.split("<script>", 1)[1].split("</script>", 1)[0], encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(path)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr


def test_browser_logic_requires_explicit_human_inputs_and_rejects_stale_import(tmp_path):
    """Run UI logic with a minimal DOM; fixture records never reach study files."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Optional Node behavioral check; no Node runtime in this environment")
    payload = {
        "bank_sha256": "explicit_engineering_fixture_only",
        "groups": [
            {
                "group_id": "fixture_group",
                "mechanism": "message",
                "prompts": [
                    {
                        "case_id": "fixture_0",
                        "prompt_sha256": "hash0",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Synthetic prompt zero; no expected answer supplied.",
                            }
                        ],
                    },
                    {
                        "case_id": "fixture_1",
                        "prompt_sha256": "hash1",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Synthetic prompt one; no expected answer supplied.",
                            }
                        ],
                    },
                ],
            }
        ],
    }
    javascript = HTML_AFTER.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = """
const assert = require('node:assert/strict');
class Element {
  constructor(){this.value='';this.textContent='';this.checked=false;this.events={};}
  addEventListener(name,fn){this.events[name]=fn;}
  append(){} replaceChildren(){} click(){}
}
const elements={};
const document={getElementById:id=>elements[id]||=(new Element()),createElement:()=>new Element()};
const storageMap=new Map();
const localStorage={getItem:key=>storageMap.get(key)||null,setItem:(key,value)=>storageMap.set(key,value),removeItem:key=>storageMap.delete(key)};
document.getElementById('study').textContent=PAYLOAD;
""".replace("PAYLOAD", json.dumps(json.dumps(payload)))
    checks = r"""
byId('decision').value='accepted';byId('decision').events.change();
byId('save').events.click();assert.equal(groupDraft().recorded_at,null);
assert.match(byId('message').textContent,/actual name/);
byId('reviewer').value='ENGINEERING FIXTURE NOT A HUMAN REVIEW';byId('reviewer').events.change();
byId('role').value='student';byId('role').events.change();
byId('read').checked=true;byId('read').events.change();
byId('save').events.click();assert.equal(groupDraft().recorded_at,null);
assert.match(byId('message').textContent,/YES\/NO answers/);
groupDraft().answers={fixture_0:'YES',fixture_1:'UNSURE'};
byId('save').events.click();assert.equal(groupDraft().recorded_at,null);
groupDraft().answers.fixture_1='NO';byId('save').events.click();
assert.equal(exportLedger().records.length,1);assert.ok(groupDraft().recorded_at);
const savedFixture=exportLedger();
assert.throws(()=>importLedger({...savedFixture,bank_sha256:'stale'}),/bank hash/);
assert.equal(exportLedger().records.length,1);
byId('notes').value='An engineering edit, not a human decision';byId('notes').events.input();
assert.equal(groupDraft().recorded_at,null);assert.throws(exportLedger,/No saved/);
savedFixture.disputed_groups=['fixture_group'];importLedger(savedFixture);
assert.deepEqual(exportLedger().disputed_groups,['fixture_group']);
console.log('Explicit-input, invalidation, stale-import and dispute-preservation checks passed.');
"""
    path = tmp_path / "review_behavior.js"
    path.write_text(harness + javascript + checks, encoding="utf-8")
    result = subprocess.run(
        [node, str(path)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr
