"""旧Word格式的离线读取门槛；合成响应不作为官方原件或真实入库证明。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_rag import official_texts as texts


def test_ole_document_is_dispatched_without_zip_parsing(monkeypatch):
    raw=b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'+b'synthetic'
    called=[]
    def legacy(value):
        called.append(value)
        return '第一条 合成正文'
    monkeypatch.setattr(texts,'read_legacy_doc',legacy,raising=False)
    assert texts.read_docx(raw)=='第一条 合成正文'
    assert called==[raw]


def test_unknown_response_is_rejected_with_format_message():
    with pytest.raises(ValueError,match='Word'):
        texts.read_docx(b'<html>not a Word document</html>')


def test_legacy_read_uses_d_drive_readonly_and_macro_disabled(tmp_path,monkeypatch):
    raw=b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'+b'synthetic'
    temp_roots=[]
    def run(command,**kwargs):
        script=command[-1]
        assert 'AutomationSecurity = 3' in script
        assert 'Documents.Open($source, $false, $true, $false)' in script
        assert 'Revisions.Count' in script
        source=Path(kwargs['env']['LEGAL_RAG_WORD_SOURCE'])
        output=Path(kwargs['env']['LEGAL_RAG_WORD_OUTPUT'])
        assert source.drive.upper()=='D:' and source.read_bytes()==raw
        temp_roots.append(source.parent)
        output.write_text('第一条 合成正文\r第二条 本解释施行。',encoding='utf-8')
        return SimpleNamespace(returncode=0,stderr=b'')
    monkeypatch.setattr(texts,'LEGACY_TEMP',tmp_path)
    monkeypatch.setattr(texts.subprocess,'run',run)
    assert texts.read_legacy_doc(raw).startswith('第一条 合成正文')
    assert len(temp_roots)==1 and not temp_roots[0].exists()
