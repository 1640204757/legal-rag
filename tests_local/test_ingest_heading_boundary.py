"""独立标题不是上一条的内容；同时防止误删条文里提到“附则”的正常句子。"""
from local_rag.ingest import extract_articles
import pytest


def test_standalone_appendix_heading_does_not_pollute_previous_article():
    """复现刑诉法正文末尾：上一条、附则标题、末条的三个独立段落。"""
    html = '<p>第一条 正文保留。</p><p>附 则</p><p>第二条 第一款。</p><p>末条第二款也须保留。</p>'
    rows = extract_articles(html, 2)
    assert rows[0]['text'] == '第一条 正文保留。'
    assert rows[1]['text'] == '第二条 第一款。\n末条第二款也须保留。'


def test_appendix_word_inside_article_is_not_removed():
    """仅跳过整行标题，不用关键词包含匹配截断合法正文。"""
    rows = extract_articles('<p>第一条 本法附则另有规定的，从其规定。</p><p>第二条 正文。</p>', 2)
    assert '本法附则另有规定的，从其规定。' in rows[0]['text']


def test_registered_footer_stops_after_all_final_article_paragraphs():
    """网页的分享按钮不属于末条；末条多款正文仍须完整保留。"""
    rows = extract_articles('<p>第一条 正文。</p><p>第二条 第一款。</p><p>第二款。</p><p>分享：</p><p>站外提示</p>', 2, footer_marker='分享：')
    assert rows[-1]['text'] == '第二条 第一款。\n第二款。'


def test_registered_footer_must_be_found_instead_of_silently_accepting_page_change():
    """来源约定的页脚消失时应隔离，防止网站改版后吸收未知尾部。"""
    with pytest.raises(ValueError, match='页脚边界'):
        extract_articles('<p>第一条 正文。</p><p>第二条 正文。</p><p>网站新页脚</p>', 2, footer_marker='分享：')


@pytest.mark.parametrize('footer, expected_status', [('分享：', 'PARSED'), ('新页脚', 'QUARANTINED')])
def test_capture_applies_source_footer_contract_and_preserves_raw(tmp_path, monkeypatch, footer, expected_status):
    """走实际采集入口，验证边界配置传到解析器；原HTML原样保存，失败产物隔离。"""
    from local_rag import ingest
    html = '<h1>测试法规</h1><p>第一条 正文。</p><p>第二条 末条正文。</p><p>'+footer+'</p><p>网站控件</p>'
    raw = html.encode('utf-8')
    monkeypatch.setattr(ingest, 'STORE', tmp_path)
    monkeypatch.setattr(ingest, 'fetch', lambda url: (raw, html))
    record = ingest.capture(dict(id='boundary-fixture', title='测试法规', last=2, footer_marker='分享：', url='https://example.invalid'))
    assert record['status'] == expected_status
    assert (tmp_path/'boundary-fixture/source.html').read_bytes() == raw
    if expected_status == 'PARSED':
        import json
        articles = json.loads((tmp_path/'boundary-fixture/articles.json').read_text(encoding='utf-8'))
        assert articles[-1]['text'] == '第二条 末条正文。'
    else:
        assert not (tmp_path/'boundary-fixture/articles.json').exists()
