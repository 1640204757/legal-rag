"""公开官方目录采集：分页发现 → 保存原文 → 条文结构检查 → 留存状态。

阅读入口：docs/guides/OFFICIAL_CATALOG_PIPELINE.md。
这里只整理资料，不生成向量。PARSED不代表法律效力认证；未施行和隔离资料不进入现行检索。
"""
import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.error import HTTPError

from .ingest import ARTICLE, ROOT, STORE, Text, chinese_number, extract_articles, fetch

CATALOG = ROOT / 'data/catalogs/official-20261020'
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'wbr'}


class Node:
    """仅供已下载HTML的结构化读取，不执行脚本、不联网。"""
    def __init__(self, tag='', attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def find(self, *, tag=None, cls=None, ident=None):
        """递归找结构元素；目录主标题和历史链接依靠不同容器区分。"""
        result = []
        if ((tag is None or self.tag == tag) and
                (cls is None or cls in self.attrs.get('class', '').split()) and
                (ident is None or self.attrs.get('id') == ident)):
            result.append(self)
        for child in self.children:
            if isinstance(child, Node):
                result.extend(child.find(tag=tag, cls=cls, ident=ident))
        return result

    def text(self):
        """保留段落换行，排除脚本/样式/导航/页脚，不改写法条文字。"""
        if self.tag in {'script', 'style', 'head', 'nav', 'footer'}:
            return ''
        content = ''.join(c.text() if isinstance(c, Node) else c for c in self.children)
        return '\n' + content + '\n' if self.tag in {'p', 'div', 'li', 'br', 'h1', 'h2', 'h3'} else content


class Tree(HTMLParser):
    """小型HTML树：仅使用标准库处理官方页面的已观察结构。"""
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        """保存元素及原有文本层级；空元素不能把后续正文包进input/meta。"""
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        """自闭合元素不改变正文容器层级。"""
        self.stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        """按最近的匹配标签回退，容忍页面无关区域的多余结束标签。"""
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        """保留字符引用解码后的正文，后续只规范空白。"""
        self.stack[-1].children.append(data)


def _official_url(base, value):
    """目录链接仅能留在当前官方站点，拒绝站外或脚本链接。"""
    url = urljoin(base, value)
    if urlsplit(url).scheme != 'https' or urlsplit(url).hostname != urlsplit(base).hostname:
        raise ValueError('目录出现非本官方站点链接')
    return url


def _date_text(nodes):
    """只接受页面明确给出的ISO日期，不从条款金额等数字猜日期。"""
    if not nodes:
        return None
    match = re.search(r'\d{4}-\d{2}-\d{2}', nodes[0].text())
    if not match:
        return None
    date.fromisoformat(match[0])
    return match[0]


def parse_moj_page(html):
    """读取司法部主目录条目与日期，排除同项内的历史版本链接。"""
    root = Tree(html).root
    totals = root.find(ident='law-total')
    if len(totals) != 1:
        raise ValueError('行政法规目录缺少唯一总数')
    total = int(totals[0].attrs['value'])
    entries = []
    for item in root.find(cls='list-item'):
        titles = item.find(cls='title')
        links = titles[0].find(tag='a') if titles else []
        if len(links) != 1:
            raise ValueError('目录主标题链接不唯一')
        entries.append(dict(provider='moj', title=links[0].text().strip(),
                            url=_official_url('https://xzfg.moj.gov.cn', links[0].attrs['href']),
                            publication_date=_date_text(item.find(cls='publish-date')),
                            effective_date=_date_text(item.find(cls='implemented-date')),
                            doc_type='ADMINISTRATIVE_REGULATION'))
    if not entries:
        raise ValueError('行政法规目录未取得任何主条目')
    return total, entries


def parse_samr_page(html):
    """读取市场监管总局公开法律目录及网页声明的总条数。"""
    root = Tree(html).root
    pagers = root.find(cls='pagination')
    if len(pagers) != 1:
        raise ValueError('法律目录分页总数缺失')
    entries = []
    for item in root.find(cls='nav04Left02_content'):
        links = item.find(tag='a')
        if len(links) != 1:
            raise ValueError('法律目录链接不唯一')
        entries.append(dict(provider='samr', title=links[0].attrs.get('title', links[0].text()).strip(),
                            url=_official_url('https://www.samr.gov.cn', links[0].attrs['href']),
                            publication_date=None, effective_date=None, doc_type='LAW'))
    if not entries:
        raise ValueError('法律目录正文为空')
    return int(pagers[0].attrs['count']), entries


def analyze_document(entry, html, as_of):
    """核验正文容器、条号及终止条款；未施行正文单列，不默认作为当前材料。"""
    if entry['provider'] not in {'moj', 'samr', 'moe'}:
        raise ValueError('未登记的正文来源，不能套用其他网站容器')
    today = date.fromisoformat(as_of)
    root = Tree(html).root
    if entry['provider'] == 'moj':
        titles = [node.text().strip() for node in root.find(cls='text-title')]
    else:
        titles = [node.attrs.get('content', '') for node in root.find(tag='meta')
                  if node.attrs.get('name') == 'ArticleTitle']
    # 名称必须绑定正文标题，不能在导航/相关链接中找到同名就认作正确资料。
    if len(titles) != 1 or re.sub(r'\s+', '', titles[0]) != re.sub(r'\s+', '', entry['title']):
        raise ValueError('正文标题与目录不匹配或不唯一')
    if entry['provider'] == 'moj':
        containers = root.find(cls='law-chapter')
        if len(containers) != 1:
            raise ValueError('行政法规正文容器缺失或不唯一')
        text = containers[0].text()
    elif entry['provider'] == 'moe':
        # 教育部实页使用独立TRS正文容器；原始施行条款不覆盖修订决定的施行日。
        containers = root.find(cls='TRS_Editor')
        if len(containers) != 1:
            raise ValueError('教育部正文容器缺失或不唯一')
        text = containers[0].text()
    else:
        containers = root.find(cls='Three_xilan_07')
        if len(containers) != 1:
            raise ValueError('市场监管总局正文容器缺失或不唯一')
        text = containers[0].text()
    lines = [re.sub(r'[ \t\u3000\xa0]+', ' ', line).strip() for line in text.splitlines()]
    matches = [ARTICLE.match(line) for line in lines]
    numbers = [chinese_number(m[1]) for m in matches if m]
    if not numbers:
        raise ValueError('未识别到条文，不能把目录或附件介绍当正文')
    last = max(numbers)
    projected = ''.join('<p>' + escape(line) + '</p>' for line in lines if line)
    articles = extract_articles(projected, last)
    tail = articles[-1]['text']
    if not re.search(r'施行|实施|生效|执行', tail):
        raise ValueError('末条未发现施行类终止条款，须人工核对全文边界')
    if re.search(r'(?m)^(?:分享[：:]|版权所有|责任编辑|网站声明|扫一扫|关闭窗口|评价建议|【E-mail)', tail):
        raise ValueError('末条包含网站控件，须登记来源正文边界')
    effective = entry.get('effective_date')
    if not effective:
        match = re.search(r'自\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日(?:起)?(?:施行|实施|生效|执行)', tail)
        if match:
            effective = date(*(int(v) for v in match.groups())).isoformat()
    future = bool(effective and date.fromisoformat(effective) > today)
    # 未能独立确认修订日期时使用快照身份，不拿网页日期伪造法律版本日期。
    version = entry.get('publication_date') or 'snapshot-' + hashlib.sha256(html.encode()).hexdigest()[:12]
    # 日期不明时只保存，不能因没有识别到未来日期就当作已经施行。
    status = 'DATE_REVIEW_REQUIRED' if not effective else 'FUTURE_EFFECTIVE' if future else 'PARSED'
    return dict(status=status, articles=articles, last=last,
                version=version, effective_date=effective, validity_status='UNVERIFIED',
                boundary_check='连续条号+末条施行类文字；未独立证明全部附件/款项层级完整')


def _save_json(path, value):
    """原子保存状态，长批次中断后可恢复，避免留下半份JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def _samr_page_url(page):
    """参数来自官网公开分页脚本，保持其默认20条每页。"""
    params = dict(parseType='bulidstatic', webId='29e9522dc89d4e088a953d8cede72f4c',
                  tplSetId='5c30fb89ae5e48b9aefe3cdf49853830', pageType='column',
                  tagId='内容区域', editType='null', pageId='e09beeae56a748989be3a8e7693e9c09',
                  paramJson=json.dumps(dict(pageNo=page, pageSize=20)))
    return 'https://www.samr.gov.cn/api-gateway/jpaas-publish-server/front/page/build/unit?' + urlencode(params)


def discover(provider):
    """遍历全部正常分页并保存目录；数量与去重未对齐时明确记为不完整。"""
    folder = CATALOG / provider
    folder.mkdir(parents=True, exist_ok=True)
    entries, pages, total, seen = [], [], None, set()
    page = 1
    while total is None or page <= math.ceil(total / (10 if provider == 'moj' else 20)):
        url = ('https://xzfg.moj.gov.cn/search2.html?PageIndex=' + str(page)) if provider == 'moj' else _samr_page_url(page)
        cached = folder / ('page-' + str(page) + '.raw')
        try:
            if cached.exists():
                raw = cached.read_bytes(); html = raw.decode('utf-8')
            else:
                raw, html = fetch(url)
                cached.write_bytes(raw)
            rendered = json.loads(html)['data']['html'] if provider == 'samr' else html
            declared, rows = parse_samr_page(rendered) if provider == 'samr' else parse_moj_page(rendered)
            if total is not None and declared != total:
                raise ValueError('分页期间目录总数变化，须重新核对')
            total = declared
            for row in rows:
                if row['url'] in seen:
                    raise ValueError('分页重复条目，不能宣称全部目录已取齐')
                seen.add(row['url']); entries.append(row)
            pages.append(dict(page=page, url=url, count=len(rows), raw_sha256=hashlib.sha256(raw).hexdigest()))
            print(provider, '目录页', page, '累计', len(entries), '/', total, flush=True)
        except Exception as exc:
            pages.append(dict(page=page, url=url, error=type(exc).__name__ + ': ' + str(exc)))
            break
        page += 1
        time.sleep(.35)
    report = dict(provider=provider, retrieved_at=datetime.now().astimezone().isoformat(),
                  declared_total=total, collected=len(entries), complete=(len(entries) == total),
                  pages=pages, entries=entries)
    _save_json(folder / 'catalog.json', report)
    return report


def collect(providers, limit=None):
    """限速采集目录正文；已存在同名资料不静默替换，失败/未来版本另存状态。"""
    known = set()
    for path in STORE.glob('*/record.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('status') == 'PARSED':
            known.add(record['title'])
    for path in (ROOT / 'data/normalized').glob('*.json'):
        known.add(json.loads(path.read_text(encoding='utf-8'))['title'])
    national = json.loads((ROOT / 'data/catalogs/national-laws-20260828/coverage.json').read_text(encoding='utf-8'))
    law_titles = {item['title'] for item in national['entries']}
    results = []
    attempted = 0
    for provider in providers:
        catalog = json.loads((CATALOG / provider / 'catalog.json').read_text(encoding='utf-8'))
        for entry in catalog['entries']:
            if provider == 'samr' and entry['title'] not in law_titles:
                results.append(dict(entry, status='OUTSIDE_NATIONAL_CATALOG_REVIEW')); continue
            ident = 'catalog-' + provider + '-' + hashlib.sha256(entry['url'].encode()).hexdigest()[:12]
            folder = STORE / ident; rp = folder / 'record.json'
            if rp.exists():
                record = json.loads(rp.read_text(encoding='utf-8'))
                results.append(record)
                if record.get('status') == 'PARSED': known.add(record['title'])
                continue
            if entry['title'] in known:
                results.append(dict(entry, status='EXISTING_TITLE_VERSION_REVIEW_REQUIRED')); continue
            if limit is not None and attempted >= limit:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            record = dict(entry, id=ident, field='GENERAL', version='UNVERIFIED',
                          validity_status='UNVERIFIED', version_applicability='UNVERIFIED',
                          reuse_permission='UNKNOWN_LOCAL_RETRIEVAL_ONLY_NOT_TRAINING_OR_REDISTRIBUTION')
            attempted += 1
            denied = False
            try:
                raw, html = fetch(entry['url'])
                (folder / 'source.html').write_bytes(raw)
                record.update(retrieved_at=datetime.now().astimezone().isoformat(), raw_sha256=hashlib.sha256(raw).hexdigest())
                parsed = analyze_document(entry, html, date.today().isoformat())
                articles = parsed.pop('articles')
                ap = folder / 'articles.json'; _save_json(ap, articles)
                record.update(parsed, articles=len(articles), parsed_sha256=hashlib.sha256(ap.read_bytes()).hexdigest())
                if record['status'] == 'PARSED': known.add(record['title'])
            except HTTPError as exc:
                record.update(status='UNAVAILABLE', error='HTTP ' + str(exc.code))
                denied = exc.code in {401, 403, 429}
            except Exception as exc:
                record.update(status='QUARANTINED' if (folder / 'source.html').exists() else 'UNAVAILABLE', error=type(exc).__name__ + ': ' + str(exc))
            _save_json(rp, record)
            results.append(record)
            _save_json(CATALOG / 'collection_report.json', dict(updated_at=datetime.now().astimezone().isoformat(), counts=dict(Counter(r['status'] for r in results)), records=results))
            print(record['title'], record['status'], record.get('articles', 0), record.get('error', ''), flush=True)
            if denied:
                print(provider, '访问被限制，停止此来源，不绕过', flush=True); break
            time.sleep(.4)
    report = dict(updated_at=datetime.now().astimezone().isoformat(), counts=dict(Counter(r['status'] for r in results)), records=results)
    _save_json(CATALOG / 'collection_report.json', report)
    return report


def main():
    """discover只保存目录；capture保存正文和产物，向量入库另用local_rag.index。"""
    parser = argparse.ArgumentParser(description='公开官方目录采集与隔离，不认证法律效力')
    parser.add_argument('action', choices=['discover', 'capture'])
    parser.add_argument('--provider', choices=['moj', 'samr', 'all'], default='all')
    parser.add_argument('--limit', type=int, help='本次最多新请求多少正文；已完成记录不重复请求')
    args = parser.parse_args()
    providers = ['samr', 'moj'] if args.provider == 'all' else [args.provider]
    if args.action == 'discover':
        for provider in providers: discover(provider)
    else:
        report = collect(providers, args.limit)
        print(json.dumps(report['counts'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
