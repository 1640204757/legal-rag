"""官方资料采集与条文切分：保存原始网页、采集时间及哈希，检查基础条号连续性。
SOURCES是明确登记的下载来源。结构检查失败的资料隔离，不能当作已入库全文。
采集成功不代表现行性、再利用许可或法律适用已经核验。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md（相对项目根目录）。
"""
import argparse
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'data/library'
SOURCES = [
    dict(id='civil-code-2020', title='中华人民共和国民法典', field='CIVIL', version='2020-05-28', last=1260,
         url='https://fgk.chinatax.gov.cn/zcfgk/c100009/c5212250/content.html'),
    dict(id='criminal-law-2023', title='中华人民共和国刑法', field='CRIMINAL', version='2023-12-29', last=452,
         url='https://fgk.chinatax.gov.cn/zcfgk/c100009/c5212248/content.html'),
    dict(id='civil-procedure-2023', title='中华人民共和国民事诉讼法', field='PROCEDURE', version='2023-09-01', last=306,
         url='https://fgk.chinatax.gov.cn/zcfgk/c100009/c5220330/content.html'),
    dict(id='admin-procedure-2017', title='中华人民共和国行政诉讼法', field='ADMINISTRATIVE', version='2017-06-27', last=103,
         url='https://www.moe.gov.cn/jyb_sjzl/sjzl_zcfg/zcfg_qtxgfl/202110/t20211029_576011.html'),
    dict(id='admin-review-2023', title='中华人民共和国行政复议法', field='ADMINISTRATIVE', version='2023-09-01', last=90,
         url='https://www.mee.gov.cn/home/ztbd/2022/sthjpf/fgbzjd/202309/t20230914_1040873.shtml'),
    dict(id='labor-contract-2012', title='中华人民共和国劳动合同法', field='LABOR', version='2012-12-28', last=98,
         url='https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/fl/202011/t20201102_394622.html'),
    dict(id='criminal-procedure-2018', title='中华人民共和国刑事诉讼法', field='PROCEDURE', version='2018-10-26', last=308,
         url='https://www.spp.gov.cn/zdgz/201810/t20181027_396818.shtml'),
    dict(id='admin-procedure-2017-tax', title='中华人民共和国行政诉讼法', field='ADMINISTRATIVE', version='2017-06-27', last=103,
         url='https://fgk.chinatax.gov.cn/zcfgk/c100009/c5211794/content.html'),
    dict(id='labor-contract-2012-samr', title='中华人民共和国劳动合同法', field='LABOR', version='2012-12-28', last=98,
         url='https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_0abfdd261c03417b949df19d869add8d.html'),
    dict(id='labor-law-2018-samr', title='中华人民共和国劳动法', field='LABOR', version='2018-12-29', last=107,
         url='https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_d9aa750028b14b99a776cb93726a360d.html'),
    dict(id='criminal-procedure-2018-mee', title='中华人民共和国刑事诉讼法', field='PROCEDURE', version='2018-10-26', last=308,
         url='https://jjjcz.mee.gov.cn/djfg/gjflfg/fl/201803/t20180311_693949.html'),
    # 核心缺口批次：版本日期取法律修订/公布日，不把网页发布日期当作修订日期。
    {'id': 'criminal-procedure-2018-samr',
     'title': '中华人民共和国刑事诉讼法',
     'field': 'PROCEDURE',
     'doc_type': 'LAW',
     'version': '2018-10-26',
     'last': 308,
     'effective_date': '2018-10-26',
     'url': 'https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2025/art_48b3b0c3f954448e996602a71fc6f878.html',
     'version_review_status': 'PUBLISHED_TEXT_IDENTIFIED_CURRENT_STATUS_UNVERIFIED',
     'version_review_as_of': '2026-10-20',
     'catalog_reference': '2026-08-28全国法律目录（法律）或来源发布文本（行政法规）；不表示之后无修订'},
    {'id': 'legal-aid-2021-moj',
     'title': '中华人民共和国法律援助法',
     'field': 'PROCEDURE',
     'doc_type': 'LAW',
     'version': '2021-08-20',
     'last': 71,
     'effective_date': '2022-01-01',
     'url': 'https://www.moj.gov.cn/pub/sfbgw/zwfw/zwfwbszn/bsznflyz/202112/t20211228_445092.html',
     'version_review_status': 'PUBLISHED_TEXT_IDENTIFIED_CURRENT_STATUS_UNVERIFIED',
     'version_review_as_of': '2026-10-20',
     'catalog_reference': '2026-08-28全国法律目录（法律）或来源发布文本（行政法规）；不表示之后无修订'},
    {'id': 'social-insurance-2018-tax',
     'title': '中华人民共和国社会保险法',
     'field': 'LABOR',
     'doc_type': 'LAW',
     'version': '2018-12-29',
     'last': 98,
     'effective_date': '2018-12-29',
     'url': 'https://fgk.chinatax.gov.cn/zcfgk/c100009/c5192937/content.html',
     'version_review_status': 'PUBLISHED_TEXT_IDENTIFIED_CURRENT_STATUS_UNVERIFIED',
     'version_review_as_of': '2026-10-20',
     'catalog_reference': '2026-08-28全国法律目录（法律）或来源发布文本（行政法规）；不表示之后无修订',
     'original_effective_date': '2011-07-01',
     'version_effective_evidence': 'https://www.sdcourt.gov.cn/jningweisfy/385991/385992/7106180/index.html'},
    {'id': 'labor-contract-implementation-2008-gov',
     'title': '中华人民共和国劳动合同法实施条例',
     'field': 'LABOR',
     'doc_type': 'ADMINISTRATIVE_REGULATION',
     'version': '2008-09-18',
     'last': 38,
     'footer_marker': '【E-mail推荐',
     'effective_date': '2008-09-18',
     'url': 'https://www.gov.cn/zwgk/2008-09/19/content_1099470.htm',
     'version_review_status': 'PUBLISHED_TEXT_IDENTIFIED_CURRENT_STATUS_UNVERIFIED',
     'version_review_as_of': '2026-10-20',
     'catalog_reference': '2026-08-28全国法律目录（法律）或来源发布文本（行政法规）；不表示之后无修订'},
    {'id': 'legal-aid-2021-beijing',
     'title': '中华人民共和国法律援助法',
     'field': 'PROCEDURE',
     'doc_type': 'LAW',
     'version': '2021-08-20',
     'last': 71,
     'footer_marker': '分享：',
     'effective_date': '2022-01-01',
     'url': 'https://www.beijing.gov.cn/zhengce/zhengcefagui/qtwj/202504/t20250402_4053713.html',
     'version_review_status': 'PUBLISHED_TEXT_IDENTIFIED_CURRENT_STATUS_UNVERIFIED',
     'version_review_as_of': '2026-10-20',
     'catalog_reference': '2026-08-28全国法律目录（法律）或来源发布文本（行政法规）；不表示之后无修订'},

]


class Text(HTMLParser):
    """HTML文本提取器：忽略脚本和页脚等内容，保留结构边界以便切分条文。"""
    def __init__(self):
        """初始化对象依赖与内部状态；后续读写通过本类的方法完成。"""
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        """处理HTML开始标签：跳过脚本、导航等内容，保留块级换行以识别条文边界。"""
        if tag in ('script','style','nav','footer','head'):
            self.skip += 1
        if not self.skip and tag in ('p','div','br','h1','h2','li','tr'):
            self.parts.append('\n')
    def handle_endtag(self, tag):
        """处理HTML结束标签，恢复跳过状态并保留段落边界。"""
        if tag in ('script','style','nav','footer','head'):
            self.skip = max(0, self.skip-1)
        if not self.skip and tag in ('p','div','h1','h2','li','tr'):
            self.parts.append('\n')
    def handle_data(self, data):
        """仅收集正文文本片段；跳过区中的内容不会加入解析文本。"""
        if not self.skip: self.parts.append(data)


def chinese_number(s):
    """将条号中的中文数字转为整数，用于检查基础条号是否连续。"""
    result = current = 0
    digits = dict(zip('零一二三四五六七八九', range(10)))
    for c in s:
        if c in digits: current = digits[c]
        else:
            unit = {'十':10,'百':100,'千':1000}[c]
            result += (current or 1)*unit
            current = 0
    return result+current


ARTICLE = re.compile(r'^第([零一二三四五六七八九十百千]+)条(之[一二三四五六七八九十]+)?[\s\u3000]+')


def extract_articles(html, last, *, footer_marker=None):
    """从网页提取条文并核对连续性；保留增补条款，不代表法律审校。

    footer_marker 是已核对来源的独立页脚行。登记后必须找到该边界，
    否则报错隔离，防止网站改版后把分享按钮等控件拼进末条。"""
    parser = Text(); parser.feed(html)
    lines = [re.sub(r'[ \t\u3000\xa0]+',' ', line).strip() for line in ''.join(parser.parts).splitlines()]
    rows = []; active = None; footer_found = False
    for number, line in enumerate(lines, 1):
        # 仅在末条后识别已登记的完整页脚行，避免截掉正文里的同词句子。
        if footer_marker and active and active['number'] == last and line == footer_marker:
            footer_found = True
            break
        match = ARTICLE.match(line)
        if match:
            n = chinese_number(match[1])
            if not rows and n != 1: continue
            if n > last: raise ValueError('正文还有后续条款，不能截断后宣称全文')
            no = '第'+match[1]+'条'+(match[2] or '')
            if any(r['article_no']==no for r in rows): raise ValueError('条号重复，须核对正文边界')
            active = dict(article_no=no, number=n, inserted=bool(match[2]), text=line, locator=f'正文提取行L{number}')
            rows.append(active)
        elif active and line:
            # 独立“附则”也是结构标题；不能拼进上一条，更不能按包含该词删正文。
            if re.match(r'^第[零一二三四五六七八九十百千]+[编章节]',line) or re.fullmatch(r'附\s*则', line):
                active = None
            elif active['number'] == last and re.match(r'^(【打印】|编辑[：:]|相关阅读|（编者注|附件[一二])', line):
                # Explicit source boundary, never assume the final article has one paragraph.
                break
            else:
                active['text'] += '\n'+line
    if footer_marker and not footer_found:
        raise ValueError('来源约定的页脚边界未出现，须复核网页结构')
    base = [r['number'] for r in rows if not r['inserted']]
    if base != list(range(1,last+1)):
        raise ValueError(f'条号不连续或全文不完整：实际{len(base)}条，预期1..{last}')
    return rows


def fetch(url):
    # Fixed public URLs only; no credentials, login, search API or CAPTCHA bypass.
    """从指定公开URL下载原始字节并严格解码；限制大小与跨站跳转，不执行网页脚本或登录。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={'User-Agent':'LegalRAG-LocalResearch/0.1'})
    with opener.open(request, timeout=40) as response:
        if response.url.split('/')[2] != url.split('/')[2]:
            raise ValueError('跨站重定向须另行核对来源')
        raw = response.read(8_000_001)
        if len(raw)>8_000_000: raise ValueError('正文超过单文件预算')
        charset = response.headers.get_content_charset()
        hint = re.search(br'charset=["\s]*([a-zA-Z0-9-]+)',raw[:5000])
        encoding = charset or (hint[1].decode() if hint else 'utf-8')
        return raw, raw.decode(encoding,errors='strict')


def capture(source):
    """保存来源网页、采集记录和条文产物；正文不匹配或条号不完整时隔离，已有记录不静默覆盖。"""
    folder = STORE/source['id']; folder.mkdir(parents=True,exist_ok=True)
    meta_path = folder/'record.json'
    if meta_path.exists(): return json.loads(meta_path.read_text(encoding='utf-8'))
    raw, html = fetch(source['url'])
    stamp=datetime.now().astimezone().isoformat(timespec='seconds')
    (folder/'source.html').write_bytes(raw)
    record=dict(source, retrieved_at=stamp, raw_sha256=hashlib.sha256(raw).hexdigest(),
                validity_status='UNVERIFIED', version_applicability='UNVERIFIED',
                reuse_permission='UNKNOWN_LOCAL_RETRIEVAL_ONLY_NOT_TRAINING_OR_REDISTRIBUTION')
    try:
        if source['title'] not in html: raise ValueError('页面标题不匹配')
        articles=extract_articles(html, source['last'], footer_marker=source.get('footer_marker'))
        (folder/'articles.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2),encoding='utf-8')
        record.update(status='PARSED', articles=len(articles),
                      parsed_sha256=hashlib.sha256((folder/'articles.json').read_bytes()).hexdigest())
    except ValueError as exc:
        record.update(status='QUARANTINED', error=str(exc))
    meta_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record


def load_extra():
    """读取结构检查通过的新增条文，验证原HTML和产物哈希后形成检索材料；隔离记录不入库。"""
    rows=[]; records=[]
    for path in sorted(STORE.glob('*/record.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['status'] != 'PARSED': continue
        # 国家库原件是DOCX，保留其真实格式；旧记录默认仍读取原HTML。
        raw_file = record.get('raw_file', 'source.html')
        if raw_file not in {'source.html', 'source.docx'}:
            raise ValueError('未登记的原始文件名，拒绝跨目录读取')
        raw=(path.parent/raw_file).read_bytes()
        article_bytes=(path.parent/'articles.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['raw_sha256'] or hashlib.sha256(article_bytes).hexdigest()!=record['parsed_sha256']:
            raise RuntimeError('新增法规的快照或条款哈希不一致：'+record['id'])
        articles=json.loads(article_bytes)
        for a in articles:
            rows.append(dict(id=record['id']+'/'+a['article_no'],kind='LAW_ARTICLE',title=record['title'],
                locator=a['article_no']+' '+a['locator'],version_id=record['version']+'（采集版本，现行性待核对）',
                text=a['text'],source_url=record['url'],content_scope='ARTICLES_ONLY',field=record['field'],
                verified_scope=('与已保存国家库Word原件对应，已校验哈希与条号连续性；仅条文，附件未入库；未独立复核现行性与个案适用' if raw_file == 'source.docx' else
                                '采集官网HTML并校验哈希与条号连续性；仅条文，附件未入库；未独立复核现行性与个案适用')))
        records.append(record)
    return rows,records


def main():
    """解析本模块命令行参数，执行对应操作；具体数据读写由下层函数负责。"""
    parser=argparse.ArgumentParser(); parser.add_argument('--ids',nargs='*'); args=parser.parse_args()
    summary=[]
    for source in SOURCES:
        if args.ids and source['id'] not in args.ids: continue
        try: record=capture(source)
        except Exception as exc: record=dict(source,status='UNAVAILABLE',error=type(exc).__name__+': '+str(exc))
        summary.append(record)
        print(source['id'],record['status'],record.get('articles',0),record.get('error',''),flush=True)
    STORE.mkdir(parents=True,exist_ok=True)
    (STORE/'batch_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
