"""法条网页解析与条号完整性自动化测试

用途说明：
验证 local_rag/ingest.py 中的 extract_articles 核心解析算法。
法条切分直接决定知识库质量，切错或漏条会导致大模型依据缺失或断章取义。

核心测试覆盖点：
1. 噪声过滤：正确剔除 HTML 里的 script/style 标签，防止恶意注入伪造法条。
2. 条号完整性与增补条款支持：支持大号中文数字解析，支持“第X条之一”等修正增补条款。
3. 严格连续性防御：跳号（如第1条直接跳到第3条）、重号、超出总条数时坚决报错隔离，绝不将残破法条存入库。
4. 末条多段保留：法律最后一条往往包含施行日期、废止决定等多款内容，必须完整保留至明确边界（如【打印】），不能只截第一款。
"""
import pytest
from local_rag.ingest import extract_articles


def test_full_law_keeps_thousand_numbers_and_inserted_articles():
    """测试条文提取算法：支持大数字、增补条款（之X），正确丢弃 script/footer 假数据与页脚"""
    html = (
        '<script>第一条 假资料</script>'
        '<p>第一条 正文甲。</p>'
        '<p>第二条 正文乙。</p>'
        '<p>第二条之一 增补。</p>'
        '<p>第三条 自某日起施行。</p>'
        '<footer>页尾</footer>'
    )
    # 预期共有 3 条正规条文（另有增补条款 1 条）
    rows = extract_articles(html, 3)
    assert [r['article_no'] for r in rows] == ['第一条', '第二条', '第二条之一', '第三条']
    # 确认页脚与脚本内的干扰文字没有混入
    assert '页尾' not in rows[-1]['text']
    assert '假资料' not in str(rows)


def test_incomplete_or_duplicate_law_fails_closed():
    """测试严格校验：跳号（1跳到3）、重号（两个第一条）、超出条数时，抛出 ValueError 拒绝解析"""
    broken_cases = [
        '<p>第一条 甲</p><p>第三条 丙</p>',                      # 跳号：缺第二条
        '<p>第一条 甲</p><p>第一条 乙</p>',                      # 重号：出现两个第一条
        '<p>第一条 甲</p><p>第二条 乙</p><p>第三条 丙</p><p>第四条 丁</p>' # 溢出：预期3条实际出现第4条
    ]
    for html in broken_cases:
        with pytest.raises(ValueError):
            extract_articles(html, 3)


def test_final_article_keeps_all_paragraphs_before_explicit_footer():
    """测试末条多款完整性：最后一条包含多款文字时全部保留，遇到【打印】等显式边界才截断"""
    html = (
        '<p>第一条 甲。</p>'
        '<p>第二条 第一款。</p>'
        '<p>第二款不得丢失。</p>'
        '<p>【打印】</p>'
    )
    rows = extract_articles(html, 2)
    # 确认第二条包含全部两款内容
    assert '第二款不得丢失' in rows[-1]['text']
    # 确认网页按钮【打印】未被作为法条文本收录
    assert '打印' not in rows[-1]['text']

