"""官方采集器边界：限定来源和跨运行访问限制，不触及外网。"""
import json
import pytest
from local_rag.official_capture import PublicClient, AccessRestricted, public_url
import urllib.request
from urllib.error import HTTPError


def test_redirected_restriction_records_actual_source(tmp_path, monkeypatch):
    """下一跳拒绝也必须持久化下一跳站点，防止恢复任务直接请求该站点。"""
    client = PublicClient(tmp_path)
    def denied(*args, **kwargs):
        raise HTTPError('https://www.npc.gov.cn/denied', 403, 'denied', {}, None)
    monkeypatch.setattr(client.opener, 'open', denied)
    with pytest.raises(AccessRestricted):
        client.fetch('https://flk.npc.gov.cn/public', tmp_path/'doc', 'source.html')
    assert 'www.npc.gov.cn' in PublicClient(tmp_path).denied


@pytest.mark.parametrize('url', ['http://flk.npc.gov.cn/x', 'https://example.com/a', 'https://user:password@flk.npc.gov.cn/a'])
def test_nonofficial_or_identified_url_rejected(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_access_restriction_survives_resume(tmp_path, monkeypatch):
    (tmp_path/'access_restrictions.json').write_text(json.dumps(dict(hosts=['flk.npc.gov.cn'])), encoding='utf-8')
    client = PublicClient(tmp_path)
    def forbidden(*args, **kwargs):
        """若没有先检查持久化限制，测试必须失败而不是访问外网。"""
        raise AssertionError('访问限制检查前不应发起请求')
    monkeypatch.setattr(client.opener, 'open', forbidden)
    with pytest.raises(AccessRestricted):
        client.fetch('https://flk.npc.gov.cn/law-search/search/flfgDetails?bbbs=example', tmp_path/'doc', 'detail.json')


def test_redirect_is_rejected_before_following(tmp_path):
    client = PublicClient(tmp_path)
    handler = next(h for h in client.opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    request = urllib.request.Request('https://flk.npc.gov.cn/public')
    with pytest.raises(ValueError):
        handler.redirect_request(request, None, 302, 'redirect', {}, 'http://example.com/untrusted')


def test_json_captcha_is_persisted_for_new_client(tmp_path, monkeypatch):
    client = PublicClient(tmp_path)
    monkeypatch.setattr(client, 'fetch', lambda *args: (b'{"code":403,"msg":"captcha required"}', {}))
    with pytest.raises(AccessRestricted):
        client.json('https://flk.npc.gov.cn/public', tmp_path/'doc', 'data.json')
    assert 'flk.npc.gov.cn' in PublicClient(tmp_path).denied


@pytest.mark.parametrize('url', ['https://fgs.moa.gov.cn/flfg/202608/t20260831_6487168.htm',
                               'https://policy.mofcom.gov.cn/claw/clawContent.shtml?id=106747'])
def test_observed_official_future_law_sources(url):
    assert public_url(url) == url
