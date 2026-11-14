"""首批原型测试：数据契约、采集时间、引用标签边界、用户链路降级。

运行：
  $env:PYTHONPATH="$PWD;$PWD/.local-deps"  # 先在项目根目录运行
  python -m pytest prototype/tests -q --rootdir prototype
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from prototype import cli
from prototype.importer import (
    LABEL_BACKGROUND,
    LABEL_STATUTORY,
    STATUTORY_LABEL_ENABLED,
    build_index,
    cn_to_int,
    coverage_statement,
    load_capture_manifest,
    load_documents,
    parse_articles,
    render_citation,
    search,
)

ROOT = Path(__file__).resolve().parent.parent.parent
NORMALIZED = ROOT / "data" / "normalized"
SCHEMA = ROOT / "schema"


@pytest.fixture(scope="module")
def docs():
    return {d.doc_id: d for d in load_documents()}


@pytest.fixture(scope="module")
def index(docs):
    return build_index(list(docs.values()))


# ---------------------------------------------------------------- 数据层


def test_chinese_numeral():
    assert cn_to_int("一") == 1
    assert cn_to_int("二十一") == 21
    assert cn_to_int("六十七") == 67


def test_parse_articles_keeps_paragraph_lines():
    arts = parse_articles(["第一条 甲。", "第二款 乙。", "第二条 丙。"], id_prefix="T")
    assert [a.article_no for a in arts] == ["第一条", "第二条"]
    assert "第二款 乙。" in arts[0].text


def test_interp2_article_count(docs):
    assert len(docs["LAW-LABOR-INTERP-2"].articles) == 21


def test_work_injury_article_count(docs):
    assert len(docs["LAW-WORK-INJURY-REGULATION"].articles) == 67


def test_first_article_locatable(docs):
    art = docs["LAW-LABOR-INTERP-2"].articles[0]
    assert art.article_no == "第一条"
    assert art.locator.startswith("L")


def test_effective_date_in_last_article(docs):
    last = docs["LAW-LABOR-INTERP-2"].articles[-1]
    assert "2025年9月1日" in last.text
    assert "第三十二条第一款同时废止" in last.text


# ---------------------------------------------------------------- 采集时间


def test_retrieved_at_comes_from_manifest(docs):
    manifest = load_capture_manifest()
    for doc_id, doc in docs.items():
        assert doc.retrieved_at == manifest[doc_id]["retrieved_at"], f"{doc_id} 采集时间被覆盖"


def test_imported_at_is_separate(docs):
    for doc in docs.values():
        assert doc.imported_at is not None
        assert doc.imported_at != doc.retrieved_at


class _MissingPath:
    def exists(self) -> bool:
        return False


class _FakeRawDir:
    """在内存中替代 data/raw，避免依赖临时目录（沙箱下 tmp_path 可能不可访问）。"""

    def __init__(self, files: dict[str, bytes]):
        self._files = files

    def __truediv__(self, name: str):
        if name not in self._files:
            return _MissingPath()
        return _FakePath(name, self._files[name])


class _FakePath:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def read_bytes(self) -> bytes:
        return self._data

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._data.decode(encoding)

    def exists(self) -> bool:
        return True


def test_missing_manifest_is_fatal(monkeypatch):
    """采集记录缺失时必须报错，不得回退到当前时间。"""
    from prototype import importer

    monkeypatch.setattr(importer, "RAW_DIR", _FakeRawDir({}))
    with pytest.raises(RuntimeError):
        importer.load_capture_manifest()


def test_hash_mismatch_is_fatal(monkeypatch):
    import hashlib

    from prototype import importer

    manifest = {
        "captures": [{
            "doc_id": "DOC", "raw_file": "x.txt", "source_url": "https://example.invalid",
            "retrieved_at": "2026-10-19T20:00:00+08:00", "retrieved_by": "test",
            "content_hash": hashlib.sha256(b"original").hexdigest(),
        }]
    }
    fake = _FakeRawDir({
        "capture_manifest.json": json.dumps(manifest).encode("utf-8"),
        "x.txt": b"tampered",
    })
    monkeypatch.setattr(importer, "RAW_DIR", fake)
    with pytest.raises(RuntimeError):
        importer.capture_for("DOC", "x.txt")


def test_matching_hash_passes(monkeypatch):
    import hashlib

    from prototype import importer

    payload = b"original"
    manifest = {
        "captures": [{
            "doc_id": "DOC", "raw_file": "x.txt", "source_url": "https://example.invalid",
            "retrieved_at": "2026-10-19T20:00:00+08:00", "retrieved_by": "test",
            "content_hash": hashlib.sha256(payload).hexdigest().upper(),
        }]
    }
    fake = _FakeRawDir({
        "capture_manifest.json": json.dumps(manifest).encode("utf-8"),
        "x.txt": payload,
    })
    monkeypatch.setattr(importer, "RAW_DIR", fake)
    capture = importer.capture_for("DOC", "x.txt")
    assert capture["retrieved_at"] == "2026-10-19T20:00:00+08:00"


# ---------------------------------------------------------------- 引用标签不可绕过


def test_statutory_label_disabled():
    assert STATUTORY_LABEL_ENABLED is False


def test_render_citation_has_no_label_param():
    assert "label" not in inspect.signature(render_citation).parameters


def test_default_label_is_background(index):
    hits = search(index, "工伤")
    assert render_citation(hits[0])["label"] == LABEL_BACKGROUND


def test_forged_eligibility_cannot_produce_statutory(index):
    hits = search(index, "工伤")
    forged = {
        "verdict": "ELIGIBLE",
        "verified_verbatim": True,
        "version_confirmed": True,
        "authority_eligible": True,
        "facts_matched": True,
    }
    assert render_citation(hits[0], eligibility=forged)["label"] == LABEL_BACKGROUND
    assert render_citation(hits[0], eligibility=forged)["label"] != LABEL_STATUTORY


def test_partial_eligibility_stays_background(index):
    hits = search(index, "工伤")
    partial = {
        "verdict": "UNVERIFIED",
        "verified_verbatim": False,
        "version_confirmed": False,
        "authority_eligible": True,
        "facts_matched": False,
    }
    assert render_citation(hits[0], eligibility=partial)["label"] == LABEL_BACKGROUND


# ---------------------------------------------------------------- 检索与相关性


def test_search_hits(index):
    hits = search(index, "工伤")
    assert hits and hits[0]["locator"]


def test_search_no_hallucination(index):
    assert search(index, "区块链智能合约") == []


def test_lexical_relevance_is_not_legal_support(index):
    hits = search(index, "工伤")
    relevance = cli.check_lexical_relevance("工伤", hits)
    assert relevance["check_type"] == "LEXICAL_RELEVANCE_ONLY"
    assert relevance["is_legal_support"] is False


def test_lexical_relevance_word_order_gap(index):
    """已知局限：词面匹配不做词序归一，'工伤认定' 仍可命中，但只是词面相关。"""
    hits = search(index, "工伤")
    relevance = cli.check_lexical_relevance("工伤认定", hits)
    assert relevance["is_legal_support"] is False


# ---------------------------------------------------------------- 用户链路（CLI 入口）


def _run(argv):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def test_cli_empty_result_degrades():
    code, out = _run(["search", "区块链智能合约"])
    assert code == 0
    assert "检索未命中" in out
    assert "degraded=True" in out


def test_cli_unverified_domain_degrades():
    code, out = _run(["search", "刑事追诉时效是多久"])
    assert code == 0
    assert "degraded=True" in out
    assert "不给出确定期限" in out


def test_cli_hit_but_unsupported_degrades():
    code, out = _run(["search", "可以 精神损害"])
    assert code == 0
    assert "degraded=True" in out
    assert ("仅有检索命中" in out) or ("无法提供法律依据" in out)
    assert "不能" in out


def test_cli_never_emits_statutory_label():
    for query in ("工伤", "工伤认定", "竞业限制"):
        code, out = _run(["search", query])
        assert code == 0
        assert LABEL_STATUTORY not in out, f"{query} 不应出现法定依据标签"
        assert LABEL_BACKGROUND in out


def test_cli_emits_coverage_statement():
    code, out = _run(["search", "工伤"])
    assert code == 0
    assert "覆盖声明" in out
    assert "案例" in out


# ---------------------------------------------------------------- Schema


@pytest.mark.parametrize(
    "name",
    ["LAW-LABOR-INTERP-2.2025-09-01.json", "LAW-WORK-INJURY-REGULATION.2010-12-20.json"],
)
def test_normalized_matches_schema(name):
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator, FormatChecker

    schema = json.loads((SCHEMA / "legal_document.schema.json").read_text(encoding="utf-8"))
    instance = json.loads((NORMALIZED / name).read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance))
    assert errors == []
