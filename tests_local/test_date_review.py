"""日期补证的失败边界：原文、版本与公告必须同时绑定，不能只凭目录日期放行。"""
import hashlib
import pytest
from local_rag.date_review import review_date


def sample():
    raw = '<meta name="ArticleTitle" content="测试法"><div class="Three_xilan_07"><p>2021年4月29日通过</p><p>第一条 正文。</p><p>第二条 本法自公布之日起施行。</p></div>'.encode()
    record = dict(title='测试法', provider='samr', status='DATE_REVIEW_REQUIRED',
                  raw_sha256=hashlib.sha256(raw).hexdigest(), publication_date=None)
    evidence = '<p>测试法2021年4月29日公布，当日起施行。</p>'.encode()
    spec = dict(title='测试法', publication_date='2021-04-29', effective_date='2021-04-29',
                body_anchor='2021年4月29日通过', evidence_anchor='测试法2021年4月29日公布，当日起施行。',
                evidence_sha256=hashlib.sha256(evidence).hexdigest(),
                evidence_url='https://www.samr.gov.cn/evidence.html')
    return record, raw, evidence, spec


def test_date_review_binds_body_version_and_evidence():
    record, raw, evidence, spec = sample()
    result, articles = review_date(record, raw, evidence, spec, '2026-10-21')
    assert result['status'] == 'PARSED'
    assert result['effective_date'] == '2021-04-29'
    assert result['validity_status'] == 'UNVERIFIED'
    assert len(articles) == 2
    assert record['status'] == 'DATE_REVIEW_REQUIRED'


@pytest.mark.parametrize('change', ['body_hash', 'evidence_hash', 'body_anchor', 'evidence_anchor', 'title', 'status'])
def test_date_review_rejects_unbound_evidence(change):
    record, raw, evidence, spec = sample()
    if change == 'body_hash': raw += b'changed'
    elif change == 'evidence_hash': evidence += b'changed'
    elif change in {'body_anchor', 'evidence_anchor', 'title'}: spec[change] = '不匹配'
    else: record['status'] = 'PARSED'
    with pytest.raises(ValueError):
        review_date(record, raw, evidence, spec, '2026-10-21')


def test_future_date_stays_isolated():
    record, raw, evidence, spec = sample()
    result, _ = review_date(record, raw, evidence, spec, '2021-04-28')
    assert result['status'] == 'FUTURE_EFFECTIVE'
