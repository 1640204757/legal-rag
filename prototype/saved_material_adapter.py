"""真实保存材料的只读适配：独立原件→条文目录，冻结trace→旁路输入。

阅读入口 build_source_catalog 和 adapt_trace。只读取已绑定字节，不联网、不解析
评测标签、不执行检索。空白归一化后的连续原文对应不代表法律适用或现行性。
"""
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile


MAX_SOURCE_BYTES = 32 * 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(root, relative, bindings, *, max_bytes=MAX_SOURCE_BYTES):
    """只读根目录内且预先绑定的文件；同一份字节用于哈希与解析。"""
    root=Path(root).resolve()
    path=(root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('读取路径越过工作区')
    name=path.relative_to(root).as_posix()
    if name not in bindings:
        raise ValueError('来源未预绑定：'+name)
    with path.open('rb') as stream:
        raw=stream.read(max_bytes+1)
    if len(raw)>max_bytes or sha(raw)!=bindings[name]:
        raise ValueError('文件超过上限或哈希不符：'+name)
    return raw


def tight(text):
    """仅删除Unicode空白；不更改标点、数字、否定词或正文。"""
    return re.sub(r'\s+','',text)


class VisibleHTML(HTMLParser):
    """离线提取HTML文本，跳过脚本/样式；不执行页面、不解析CSS布局。"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored=0
        self.parts=[]

    def handle_starttag(self, tag, attrs):
        if tag in ('script','style'):
            self.ignored+=1

    def handle_endtag(self, tag):
        if tag in ('script','style') and self.ignored:
            self.ignored-=1

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def source_text(raw, filename):
    """仅支持UTF-8 HTML和DOCX正文XML；不启动浏览器/Office，不做旧DOC转换。"""
    if filename=='source.html':
        parser=VisibleHTML()
        parser.feed(raw.decode('utf-8-sig'))
        return '\n'.join(parser.parts)
    if filename=='source.docx':
        with zipfile.ZipFile(io.BytesIO(raw)) as stream:
            info=stream.getinfo('word/document.xml')
            if info.file_size>MAX_SOURCE_BYTES:
                raise ValueError('DOCX正文XML超过上限')
            tree=ET.fromstring(stream.read(info))
            tag='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'
            return '\n'.join(node.text or '' for node in tree.iter(tag))
    raise ValueError('未知原始文件格式，拒绝自动转换')


def build_source_catalog(root, source_id, bindings):
    """由独立record/raw/articles生成完整法条目录；正文不从检索候选取值。

    限制：只支持data/library已PARSED规范材料，不处理案例摘要或外部URL抓取。
    bindings须在执行前固定；其本身不独立认证官方记录的真实性。
    """
    if not isinstance(source_id,str) or source_id in ('.','..') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',source_id):
        raise ValueError('不安全的来源ID')
    folder=Path('data/library')/source_id
    record_raw=read_bound(root,folder/'record.json',bindings)
    record=json.loads(record_raw)
    url=record.get('url','')
    parsed=urlparse(url)
    host=parsed.hostname or ''
    if record.get('id')!=source_id or record.get('status')!='PARSED' or parsed.scheme not in ('http','https') or not (host=='gov.cn' or host.endswith('.gov.cn')) or parsed.username or parsed.password:
        raise ValueError('来源身份/状态或登记官方域名无效：'+source_id)
    for field in ('title','version','raw_sha256','parsed_sha256'):
        if not isinstance(record.get(field),str) or not record[field].strip():
            raise ValueError('采集字段缺失：'+field)
    filename=record.get('raw_file','source.html')
    if filename not in ('source.html','source.docx'):
        raise ValueError('未知原始文件名')
    raw=read_bound(root,folder/filename,bindings)
    article_raw=read_bound(root,folder/'articles.json',bindings)
    if sha(raw).lower()!=record['raw_sha256'].lower() or sha(article_raw).lower()!=record['parsed_sha256'].lower():
        raise ValueError('采集记录与原件/条文文件哈希不符：'+source_id)
    articles=json.loads(article_raw)
    if not isinstance(articles,list) or not articles or len(articles)!=record.get('articles'):
        raise ValueError('条文列表/数量与采集记录不符：'+source_id)
    plain=tight(source_text(raw,filename))
    rows=[]; seen=set(); matches=[]
    for article in articles:
        number=article.get('article_no') if isinstance(article,dict) else None
        text=article.get('text') if isinstance(article,dict) else None
        if not isinstance(number,str) or not re.fullmatch(r'第[〇零一二三四五六七八九十百千万两0-9]+条(?:之[〇零一二三四五六七八九十百千万两0-9]+)?',number) or number in seen or not isinstance(text,str) or not text.strip():
            raise ValueError('条号重复/缺失或正文无效：'+source_id)
        normalized=tight(text)
        if not normalized.startswith(tight(number)) or normalized not in plain:
            raise ValueError('条文未在保存原文中连续对应：'+source_id+'/'+number)
        seen.add(number)
        rows.append(dict(id=source_id+'/'+number,title=record['title'],text=text,
                         source_url=url,version_id=record['version']+'（采集版本，现行性待核对）'))
        matches.append(dict(id=source_id+'/'+number,text_sha256=sha(text.encode()),raw_text_match='PASS_WHITESPACE_ONLY'))
    provenance=dict(source_id=source_id,title=record['title'],source_url=url,raw_format=filename,
                    article_count=len(rows),raw_text_match_count=len(matches),normalization='REMOVE_WHITESPACE_ONLY',
                    version_applicability='NOT_TESTED',current_validity='NOT_TESTED',
                    fetched_at=record.get('retrieved_at'),reuse_permission=record.get('reuse_permission','UNKNOWN'),
                    files_sha256={p.as_posix():bindings[p.as_posix()] for p in (folder/'record.json',folder/filename,folder/'articles.json')},
                    articles=matches)
    return rows,provenance


def adapt_trace(trace, corpus):
    """从保存的实际trace恢复有序候选；无标签和重排，真实答案引文缺失时为空。

    完整trace按值保留；输入corpus必须另经冻结哈希验证，不用它生成独立来源目录。
    """
    ids=trace.get('a_ordered_ids')
    if trace.get('complete') is not True or not isinstance(ids,list) or not ids or len(ids)>12 or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('trace不完整或A候选ID无效')
    if trace.get('k4')!=ids[:4] or trace.get('k8')!=ids[:8]:
        raise ValueError('trace的前四/前八不是原候选前缀')
    pool=trace.get('fused_pool')
    if not isinstance(pool,list) or [row.get('id') for row in pool[:len(ids)]]!=ids:
        raise ValueError('trace的A候选与保存全池顺序不符')
    by_id={row['id']:row for row in corpus}
    if len(by_id)!=len(corpus) or any(i not in by_id for i in ids):
        raise ValueError('语料ID重复或缺少trace候选')
    query=trace.get('query')
    if not isinstance(query,str) or not query.strip():
        raise ValueError('trace缺少原始问题')
    return dict(question=query,candidates=[deepcopy(by_id[i]) for i in ids],citations=[],
                semantic_observations=[],actions=[],branch_trace=deepcopy(trace),
                citation_origin='NO_GENERATED_ANSWER_IN_SAVED_TRACE',
                data_use='REVEALED_SNAPSHOT_ENGINEERING_CHECK_ONLY')
