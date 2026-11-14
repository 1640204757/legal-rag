"""官方正文适配回归：DOCX整段顺序、条号边界、标题/日期及来源容器。"""
from html import escape
from io import BytesIO
import zipfile
import pytest
from local_rag.official_texts import read_docx, analyze_plain, analyze_court


def docx(lines):
    """构造最小Word正文，验证分开的条号与正文runs能按段落读取。"""
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        content = ''.join('<w:p><w:r><w:t>'+escape(line)+'</w:t></w:r></w:p>' for line in lines)
        z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+content+'</w:body></w:document>')
    return stream.getvalue()


ENTRY = dict(title='中华人民共和国示例法', publication_date='2020-01-01', effective_date='2020-02-01')


def test_docx_paragraphs_and_standalone_article_heading():
    text = read_docx(docx([ENTRY['title'], '第一条', '为了示例，制定本法。', '第二条', '本法自2020年2月1日起施行。']))
    parsed = analyze_plain(ENTRY, text, '2026-10-21')
    assert parsed['last'] == 2 and parsed['status'] == 'PARSED'
    assert '为了示例' in parsed['articles'][0]['text']


@pytest.mark.parametrize('text', ['错误法名\n第一条 内容\n第二条 本法施行。',
                                  '关于修改《'+ENTRY['title']+'》的决定\n第一条 修改某条。\n第二条 本决定施行。',
                                  ENTRY['title']+'\n第一条 内容\n第三条 本法施行。',
                                  ENTRY['title']+'\n第一条 目录摘要\n第二条 尚有后续正文'])
def test_wrong_title_gap_or_unfinished_tail_is_rejected(text):
    with pytest.raises(ValueError):
        analyze_plain(ENTRY, text, '2026-10-21')


def test_future_effective_is_not_current():
    parsed = analyze_plain(dict(ENTRY, effective_date='2027-01-01'), ENTRY['title']+'\n第一条 本法自2027年1月1日起施行。', '2026-10-21')
    assert parsed['status'] == 'FUTURE_EFFECTIVE'


def test_court_body_does_not_take_related_news():
    entry = dict(ENTRY, title='最高人民法院关于示例的解释', provider='cicc')
    html = '<article class="essay-content"><p>'+entry['title']+'</p><p>第一条 示例。</p><p>第二条 本解释施行。</p></article><nav>第三条 假正文</nav>'
    assert analyze_court(entry, html, '2026-10-21')['last'] == 2


def test_court_without_known_container_is_rejected():
    with pytest.raises(ValueError):
        analyze_court(dict(ENTRY, provider='cicc'), '<p>第一条 示例施行</p>', '2026-10-21')


def test_word_tabs_do_not_glue_article_number_to_text():
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>第一条</w:t><w:tab/><w:t>示例</w:t></w:r></w:p></w:body></w:document>')
    assert read_docx(stream.getvalue()) == '第一条 示例'


def test_legacy_court_txt_container_is_separate_from_navigation():
    entry = dict(ENTRY, provider='court')
    html = '<div class="txt_txt"><p>'+ENTRY['title']+'</p><p>第一条 示例。</p><p>第二条 本法施行。</p></div><nav>第三条 伪条文</nav>'
    assert analyze_court(entry, html, '2026-10-21')['last'] == 2
