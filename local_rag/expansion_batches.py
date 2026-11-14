"""暂存正文的分批发布：整份分组、同名去重、原件与重解析审计。

主入口group_records、audit_record、publish_record；find_catalog_updates仅在内存盘点目录缺口与新版。
盘点不下载、不发布、不写向量库，与在线问答兜底分工独立。
已隔离记录不能通过本模块混入当前库；重复来源同一法名不能自动当作不同法律。
"""
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from .batch_storage import publish_document
from .official_texts import analyze_plain, analyze_court, read_docx
from .official_capture import catalog_date, catalog_entry_metadata, normalize_catalog_title


def find_catalog_updates(catalog_entries, indexed_records):
    """输入官方目录及本地记录，返回缺失/明确更新的安全元数据列表；无网络或磁盘副作用。

    同名本地多条记录按各日期最新值比较；本地日期未知时不推定官方更晚，也不认证已是最新。
    日期不做法定生效判断，本函数不能直接触发下载、发布或向量重建。
    """
    indexed = {}
    for record in indexed_records:
        title = normalize_catalog_title(record.get('title'))
        if not title:
            continue
        dates = indexed.setdefault(title, {'publication_date': '', 'effective_date': ''})
        publication = catalog_date(record.get('publication_date') or record.get('gbrq') or record.get('version') or record.get('version_id'))
        effective = catalog_date(record.get('effective_date') or record.get('sxrq'))
        dates['publication_date'] = max(dates['publication_date'], publication)
        dates['effective_date'] = max(dates['effective_date'], effective)
    updates, seen = [], set()
    for entry in catalog_entries:
        metadata = catalog_entry_metadata(entry)
        if metadata is None:
            continue
        title = normalize_catalog_title(metadata['title'])
        local = indexed.get(title)
        if local is None:
            update_type = 'MISSING_FROM_LOCAL'
        elif any(metadata[key] and local[key] and metadata[key] > local[key]
                 for key in ('publication_date', 'effective_date')):
            update_type = 'VERSION_UPDATED_ON_OFFICIAL'
        else:
            continue
        identity = (title, metadata['official_url'], metadata['publication_date'], metadata['effective_date'])
        if identity not in seen:
            seen.add(identity)
            updates.append(dict(metadata, update_type=update_type))
    return updates


def verify_cached_baseline(entries, keys):
    """逐键验证旧基线向量；新增向量再多也不能掩盖一个旧条文缺失或损坏。"""
    for key in keys:
        vector = entries.get(key)
        if not isinstance(vector, list) or len(vector) != 1024 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError('旧基线向量缺失或无效，未自动重算：'+key)
    return len(keys)


def group_records(records, budget):
    """按条文预算整份分组并排除隔离/重复法名；超预算单份独占一批，不拆断法规。"""
    if budget <= 0:
        raise ValueError('批次预算必须大于零')
    groups, current, count, seen = [], [], 0, set()
    for record in records:
        if record['status'] != 'PARSED' or record['title'] in seen:
            continue
        seen.add(record['title'])
        if current and count + record['articles'] > budget:
            groups.append(current)
            current, count = [], 0
        current.append(record)
        count += record['articles']
    if current:
        groups.append(current)
    return groups


def audit_record(folder):
    """校验原始字节、派生产物及重解析完全一致；只接受已登记的原件格式和PARSED状态。"""
    folder = Path(folder)
    record = json.loads((folder/'record.json').read_text(encoding='utf-8'))
    name = record.get('raw_file', 'source.html')
    if record['status'] != 'PARSED' or name not in {'source.html', 'source.docx'}:
        raise ValueError('暂存记录未通过结构/日期检查或原件名未登记')
    raw, body = (folder/name).read_bytes(), (folder/'articles.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['raw_sha256'] or hashlib.sha256(body).hexdigest() != record['parsed_sha256']:
        raise ValueError('正文原件或派生条文哈希不一致')
    parsed = analyze_plain(record, read_docx(raw), date.today().isoformat()) if name == 'source.docx' else analyze_court(record, raw.decode('utf-8'), date.today().isoformat())
    if parsed['status'] != 'PARSED' or parsed['articles'] != json.loads(body):
        raise ValueError('重解析与暂存条文不一致')
    return record, {name: raw, 'articles.json': body, 'record.json': (folder/'record.json').read_bytes()}


def publish_record(folder, library, staging):
    """完整审计后原子发布一份记录，未知同目录内容拒绝覆盖；不修改原始快照。"""
    record, files = audit_record(folder)
    publish_document(Path(library)/record['id'], Path(staging)/record['id'], files)
    return record
