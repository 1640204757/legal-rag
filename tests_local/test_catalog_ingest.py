"""官方大目录采集、解析与未生效法规隔离机制自动化测试

用途说明：
验证 local_rag/catalog_ingest.py 中的大批量法规采集与解析流水线逻辑。
包含司法部（MOJ）606项行政法规目录与国家市场监督管理总局（SAMR）118项法律目录的解析测试。

核心测试覆盖点：
1. 目录分页与元数据抽取：正确读取公布日期与施行日期，剔除历史旧版折叠链接。
2. 总数核验：页面如果缺失官方声明的总数，直接报错拒绝，防止因网络中断只抓了半页就宣称全部完成。
3. 容器边界截断：精准匹配正文容器，剔除容器外的页眉、页脚、分享按钮和联系我们等噪声。
4. 未生效法规严格隔离（FUTURE_EFFECTIVE）：
   - 如果一部法律已经通过但施行日期在未来（例如2026年12月1日才施行），系统会自动将其状态标记为 FUTURE_EFFECTIVE；
   - 这种法规只保存备份，【绝对不进入】当前的向量知识库，防止给用户提供尚未生效的依据。
5. 实施日期不明待核验（DATE_REVIEW_REQUIRED）：条文写“公布之日起施行”但没有具体日期的，标记待复核，不盲目宣称现行有效。
"""
import pytest
from local_rag.catalog_ingest import parse_moj_page, parse_samr_page, analyze_document


def test_moe_body_uses_observed_container_and_excludes_navigation():
    html = ('<meta name="ArticleTitle" content="测试教育法">'
            '<div class="TRS_Editor"><p>第一条 正文。</p><p>第二条 本法自1995年9月1日起施行。</p></div>'
            '<footer>网页日期2021-07-30 第三条 相关链接</footer>')
    result = analyze_document(dict(provider='moe', title='测试教育法',
                                  publication_date='2021-04-29', effective_date='2021-04-30'),
                              html, '2026-10-21')
    assert result['status'] == 'PARSED' and result['last'] == 2
    assert result['effective_date'] == '2021-04-30'
    assert '网页日期' not in str(result['articles'])


@pytest.mark.parametrize('html', [
    '<meta name="ArticleTitle" content="其他法"><div class="TRS_Editor"><p>第一条 正文自公布之日起施行。</p></div>',
    '<meta name="ArticleTitle" content="测试教育法"><nav>第一条 正文自公布之日起施行。</nav>',
    '<meta name="ArticleTitle" content="测试教育法"><div class="TRS_Editor">正文摘要</div>',
    '<meta name="ArticleTitle" content="测试教育法"><div class="TRS_Editor"><p>第一条 正文自公布之日起施行。</p></div><div class="TRS_Editor">重复容器</div>',
])
def test_moe_rejects_wrong_title_missing_or_ambiguous_body(html):
    with pytest.raises(ValueError):
        analyze_document(dict(provider='moe', title='测试教育法'), html, '2026-10-21')


def test_moj_directory_ignores_history_links_and_retains_dates():
    """测试司法部目录解析：正确提取总数（606）、现行链接与施行日期，忽略 fold-con 折叠层里的历史修订版"""
    html = (
        '<input id="law-total" value="606">'
        '<li class="list-item">'
        '<div class="title"><a href="https://xzfg.moj.gov.cn/front/law/detail?LawID=99">测试条例</a></div>'
        '<ul class="info-list">'
        '<li class="publish-date">2026-09-19公布</li>'
        '<li class="implemented-date">2026-12-01施行</li>'
        '</ul>'
        '<div class="fold-con"><a href="https://xzfg.moj.gov.cn/front/law/detail?LawID=11">测试条例旧版</a></div>'
        '</li>'
    )
    total, rows = parse_moj_page(html)
    assert total == 606 and len(rows) == 1
    # 确认提取的是主条目的现行链接，而不是折叠旧版
    assert rows[0]['url'].endswith('LawID=99')
    assert rows[0]['effective_date'] == '2026-12-01'


def test_samr_directory_reads_declared_total_and_official_url():
    """测试市监局目录解析：读取声明总数（118）及标题与URL"""
    html = (
        '<li class="nav04Left02_content"><a href="/x.html" title="测试法">测试法</a></li>'
        '<div class="pagination" count="118"></div>'
    )
    total, rows = parse_samr_page(html)
    assert total == 118 and rows[0]['url'] == 'https://www.samr.gov.cn/x.html'


def test_incomplete_directory_cannot_claim_total_coverage():
    """测试总数防缺口机制：目录页面如果没有声明总数，抛出 ValueError，绝不盲目假设已全量抓取"""
    with pytest.raises(ValueError):
        parse_moj_page('<li class="list-item">缺少总数</li>')


def fixture_html(body):
    """构建标准法条 HTML 测试骨架，正文外故意附带 footer 网站控件，用于测试边界过滤"""
    return (
        '<div class="text-title">测试条例</div>'
        '<div class="law-chapter">' + body + '</div>'
        '<footer>网站控件</footer>'
    )


def test_document_uses_body_container_and_preserves_final_paragraphs():
    """测试正文提取：正文必须完整提取至末条第二款，且容器外的【网站控件】绝不能被吸收到条文中"""
    result = analyze_document(
        dict(title='测试条例', provider='moj', publication_date='2020-01-01', effective_date='2020-02-01'),
        fixture_html('<p>第一条 正文。</p><p>附则</p><p>第二条 本条例自2020年2月1日起施行。</p><p>旧规定同时废止。</p>'),
        '2026-10-20'
    )
    assert result['status'] == 'PARSED' and result['last'] == 2
    # 确认第二款内容没有被截断丢弃
    assert '旧规定同时废止。' in result['articles'][-1]['text']
    # 确认外层的网站控件已被剔除
    assert '网站控件' not in str(result['articles'])


def test_future_effective_text_is_saved_but_not_admitted_as_parsed():
    """测试未来生效法规隔离机制：施行日期在基准日期之后的法规，标记为 FUTURE_EFFECTIVE，不进入可用知识库"""
    result = analyze_document(
        dict(title='测试条例', provider='moj', publication_date='2026-09-19', effective_date='2026-12-01'),
        fixture_html('<p>第一条 正文。</p><p>第二条 本条例自2026年12月1日起施行。</p>'),
        '2026-10-20'  # 当前核验基准日期（早于2026-12-01）
    )
    # 状态必须是 FUTURE_EFFECTIVE 而非 PARSED
    assert result['status'] == 'FUTURE_EFFECTIVE'


@pytest.mark.parametrize('body', [
    '<p>第一条 正文。</p><p>第三条 本条例自公布之日起施行。</p>',  # 跳号
    '<p>第一条 正文。</p><p>第一条 重复。</p>',                    # 重号
    '<p>第一条 只取得前半部分。</p>',                             # 严重残缺
])
def test_invalid_or_truncated_text_is_rejected(body):
    """测试异常排版坚决拒绝：条号不连续或截断时抛出 ValueError，不强行入库"""
    with pytest.raises(ValueError):
        analyze_document(dict(title='测试条例', provider='moj'), fixture_html(body), '2026-10-20')


def test_title_mismatch_is_rejected():
    """测试标题校验：页面标题与预期不符时抛出异常，防止抓错或抓到重定向页面"""
    with pytest.raises(ValueError):
        analyze_document(dict(title='另一部条例', provider='moj'), fixture_html('<p>第一条 正文。</p><p>第二条 本条例自公布之日起施行。</p>'), '2026-10-20')


def test_title_appearing_only_in_related_link_is_not_accepted():
    """测试标题位置假阳性防御：标题如果只是出现在网页相关的超链接文字里，而非正文主标题，坚决拒绝"""
    html = (
        '<div class="text-title">其他条例</div>'
        '<a>测试条例</a>'  # 仅在普通 a 标签里提到
        '<div class="law-chapter"><p>第一条 正文。</p><p>第二条 本条例自公布之日起施行。</p></div>'
    )
    with pytest.raises(ValueError):
        analyze_document(dict(title='测试条例', provider='moj'), html, '2026-10-20')


@pytest.mark.parametrize('verb', ['实施', '生效', '执行'])
def test_future_date_with_other_effective_verbs_is_isolated(verb):
    """测试多关键词未来生效识别：支持识别'自X日起实施/生效/执行'等各种表述，均判定为 FUTURE_EFFECTIVE"""
    html = (
        '<meta name="ArticleTitle" content="测试法">'
        f'<div class="Three_xilan_07"><p>第一条 正文。</p><p>第二条 本法自2026年12月1日起{verb}。</p></div>'
    )
    result = analyze_document(dict(title='测试法', provider='samr'), html, '2026-10-20')
    assert result['status'] == 'FUTURE_EFFECTIVE'


def test_samr_body_excludes_unregistered_outside_page_text():
    """测试市监局页面外层过滤：正文容器之外的“联系我们”等尾部控件必须被剔除"""
    html = (
        '<meta name="ArticleTitle" content="测试法">'
        '<div class="Three_xilan_07"><p>第一条 正文。</p><p>第二条 本法自2020年1月1日起施行。</p></div>'
        '<div>联系我们</div>'
    )
    result = analyze_document(dict(title='测试法', provider='samr'), html, '2026-10-20')
    assert '联系我们' not in result['articles'][-1]['text']


def test_samr_missing_body_container_is_rejected():
    """测试市监局结构完整性：如果页面找不到标准正文容器（Three_xilan_07），抛出异常隔离"""
    html = '<meta name="ArticleTitle" content="测试法"><p>第一条 正文。</p><p>第二条 本法自公布之日起施行。</p>'
    with pytest.raises(ValueError):
        analyze_document(dict(title='测试法', provider='samr'), html, '2026-10-20')


def test_unknown_effective_date_is_not_admitted():
    """测试施行日期不明待核验：只写了'公布之日起施行'但元数据没有确切日期的，标记 DATE_REVIEW_REQUIRED"""
    html = fixture_html('<p>第一条 正文。</p><p>第二条 本条例自公布之日起施行。</p>')
    result = analyze_document(dict(title='测试条例', provider='moj'), html, '2026-10-21')
    assert result['status'] == 'DATE_REVIEW_REQUIRED'
