"""新增官方正文的离线解析：国家库DOCX及最高法独立正文容器。

阅读入口：read_docx → analyze_plain；网页使用analyze_court。输入已保存原始字节，
输出条文结构及日期状态；DOCX离线解析，旧DOC经本机Word只读投影到D盘临时文件，不联网。
原始Word保留不改，XML段落投影只是派生产物，不冒充官网下载HTML。
"""
from datetime import date
from contextlib import contextmanager
from html import escape
from io import BytesIO
import re
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import zipfile
import xml.etree.ElementTree as ET

from .catalog_ingest import Tree
from .ingest import chinese_number, extract_articles

HEADING = re.compile(r'^第([零一二三四五六七八九十百千]+)条(之[一二三四五六七八九十]+)?(?=\s|$)')
LEGACY_TEMP=Path(__file__).resolve().parents[1]/'artifacts/local-rag/legacy-word-temp'


@contextmanager
def _legacy_temp_folder():
    """用父目录继承权限创建唯一工作目录，退出时只清理本次目录。"""
    parent = LEGACY_TEMP.resolve()
    for _ in range(5):
        folder = parent / f'readonly-{uuid.uuid4().hex}'
        try:
            # 不用TemporaryDirectory的POSIX 0700模式；Windows受限ACL下可能连清理枚举都被拒。
            folder.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError('无法分配唯一旧Word临时目录')
    try:
        yield folder
    finally:
        # 与原实现保持一致：清理失败不覆盖解析结果；目录名随机且限定在LEGACY_TEMP内。
        if folder.parent == parent and folder.name.startswith('readonly-'):
            shutil.rmtree(folder, ignore_errors=True)


def read_legacy_doc(raw):
    """用已安装Word读取OLE/DOC，禁宏、禁链接更新、只读且不保存；原件字节不改变。

    副作用：创建隐藏Word实例及D盘临时原件副本/文本投影，正常结束即清理临时副本。
    不安装Office；无法打开、存在修订或超时均报错，不凭字符扫描拼凑旧Word正文。
    """
    if os.name!='nt' or LEGACY_TEMP.drive.upper()!='D:':
        raise ValueError('旧Word读取需要Windows及D盘临时目录')
    LEGACY_TEMP.mkdir(parents=True,exist_ok=True)
    script=r'''
$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
$previousLinks = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $previousLinks = $word.Options.UpdateLinksAtOpen
    $word.Options.UpdateLinksAtOpen = $false
    $source = $env:LEGAL_RAG_WORD_SOURCE
    $doc = $word.Documents.Open($source, $false, $true, $false)
    if ($doc.Revisions.Count -gt 0) { throw 'Word存在未裁定修订，不能自动选取版本' }
    [IO.File]::WriteAllText($env:LEGAL_RAG_WORD_OUTPUT, $doc.Content.Text, (New-Object Text.UTF8Encoding($false)))
} finally {
    if ($null -ne $doc) { $doc.Close(0); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) }
    if ($null -ne $word) {
        if ($null -ne $previousLinks) { $word.Options.UpdateLinksAtOpen = $previousLinks }
        $word.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
}
'''
    # Office诊断日志可能短暂占用文件；目录继承D盘父目录权限，避免受限ACL阻断清理。
    with _legacy_temp_folder() as folder:
        source=Path(folder)/'source.doc'
        output=Path(folder)/'text.txt'
        source.write_bytes(raw)
        env=dict(os.environ,LEGAL_RAG_WORD_SOURCE=str(source),LEGAL_RAG_WORD_OUTPUT=str(output),
                 TEMP=folder,TMP=folder)
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
                              env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode or not output.exists():
            raise ValueError('旧Word只读提取失败；需本机Microsoft Word且原件无修订/加密')
        if source.read_bytes()!=raw:
            raise ValueError('Word读取期间临时原件副本改变，拒绝解析')
        text=output.read_text(encoding='utf-8').replace('\x07','\n')
    if not text.strip():
        raise ValueError('旧Word正文为空')
    return text


def read_docx(raw):
    """按真实魔数读取Word，不信任下载参数/扩展名；拒绝网页伪装为Word。"""
    if raw.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
        return read_legacy_doc(raw)
    if not zipfile.is_zipfile(BytesIO(raw)):
        raise ValueError('响应不是可识别的Word文档，保留原件后停止')
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        part = archive.getinfo('word/document.xml')
        if part.file_size > 12_000_000:
            raise ValueError('Word正文超过大小预算')
        root = ET.fromstring(archive.read(part))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    if root.findall('.//w:del', ns) or root.findall('.//w:ins', ns):
        raise ValueError('Word存在未裁定修订痕迹，不能自动选取版本')
    # 不能只拼接w:t：制表符常用于分开条号与正文，丢失会把“第一条”为了粘在一起。
    lines = []
    for para in root.findall('.//w:p', ns):
        parts = []
        for node in para.iter():
            name = node.tag.rsplit('}', 1)[-1]
            if name == 't':
                parts.append(node.text or '')
            elif name in {'tab', 'br', 'cr'}:
                parts.append(' ' if name == 'tab' else '\n')
        lines.append(''.join(parts))
    if not any(line.strip() for line in lines):
        raise ValueError('Word正文为空')
    return '\n'.join(lines)


def analyze_plain(entry, text, as_of):
    """核验标题、连续条号、末条边界和官方日期，返回PARSED或日期隔离状态。

    只处理逐条正文；没有编号、目录、标题不匹配及末条缺少施行类文字均拒绝。
    附件/表格完整性仍未认证，采集内容不能因末条连续而宣称完整法律效力核验。
    """
    lines = [re.sub(r'[ \t\u3000\xa0]+', ' ', line).strip() for line in text.splitlines() if line.strip()]
    normalize = lambda value: re.sub(r'\s+', '', value)
    title = normalize(entry['title'])
    head = [normalize(line) for line in lines[:10]]
    windows = [''.join(head[i:i+length]) for i in range(len(head)) for length in (1, 2, 3)]
    # 标题必须是独立主标题（允许分行及修正年份后缀），不能只在“关于修改《某法》的决定”中找到法名。
    if not any(value == title or value.startswith(title+'（') or value.startswith(title+'(') for value in windows):
        raise ValueError('正文前部标题与官方目录不匹配')
    prepared = []
    for line in lines:
        match = HEADING.match(line)
        # Word常将“第一条”和条文内容拆成两个独立段落，显式合并后再交给原有解析器。
        if prepared and HEADING.fullmatch(prepared[-1]):
            if match:
                raise ValueError('条号后没有条文内容')
            prepared[-1] += ' ' + line
        else:
            prepared.append(line)
    numbers = [chinese_number(m[1]) for line in prepared if (m := HEADING.match(line))]
    if not numbers:
        raise ValueError('未发现逐条正文，不能将目录当作正文')
    last = max(numbers)
    articles = extract_articles(''.join('<p>'+escape(line)+'</p>' for line in prepared), last)
    if not re.search('施行|实施|生效|执行', articles[-1]['text']):
        raise ValueError('末条没有已知施行边界，需要单独核查')
    if re.search(r'(?m)^(?:责任编辑|相关链接|版权所有|分享|扫一扫)', articles[-1]['text']):
        raise ValueError('末条混入网页控件')
    effective = entry.get('effective_date')
    status = 'DATE_REVIEW_REQUIRED' if not effective else 'FUTURE_EFFECTIVE' if date.fromisoformat(effective) > date.fromisoformat(as_of) else 'PARSED'
    return dict(status=status, articles=articles, last=last, effective_date=effective,
                version=entry.get('publication_date') or 'VERSION_REVIEW_REQUIRED', validity_status='UNVERIFIED',
                boundary_check='正文标题、连续条号及末条施行文字；附件与个案适用未认证')


def analyze_court(entry, html, as_of):
    """仅解析已观察到的最高法正文容器；不将新闻稿、相关链接混作司法解释。"""
    root = Tree(html).root
    nodes = root.find(tag='article', cls='essay-content') if entry['provider'] == 'cicc' else root.find(ident='zoom')
    if not nodes and entry['provider'] == 'court':
        # 已观察的最高法旧版模板只有txt_txt类，无zoom标识；仍要求唯一独立正文。
        nodes = root.find(cls='txt_txt')
    if len(nodes) != 1:
        raise ValueError('最高法独立正文容器缺失或不唯一')
    text = nodes[0].text()
    # 最高法发布会页附带解释全文与相关链接，边界必须是已观察的独立“相关链接：”行。
    text = re.split(r'(?m)^\s*相关链接[：:]\s*$', text)[0]
    if entry['provider'] != 'cicc':
        starts = list(re.finditer(r'(?m)^\s*最高人民法院\s*$', text))
        if starts:
            text = text[starts[-1].start():]
    return analyze_plain(entry, text, as_of)
