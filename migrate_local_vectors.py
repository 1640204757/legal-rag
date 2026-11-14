"""旧版本地向量缓存格式迁移工具

用途：
- 本工具为一次性数据格式升级脚本，日常运行问答和资料入库时【无需使用】。
- 功能：将早期版本无条文哈希键的纯列表式向量缓存，转换为以条文唯一指纹为 Key 的字典缓存结构。
- 安全机制：
  1. 必须通过模型与旧语料指纹比对（fingerprint），不匹配则直接拒绝迁移，防止向量错位。
  2. 自动在同目录下备份原始缓存为 embeddings.v1.backup.json。
  3. 使用临时文件原子替换，防止迁移中途断电损坏原文件。
"""
import hashlib
import json
import asyncio
from local_rag.retrieval import ROOT, load_corpus
from local_rag.ollama import OllamaClient


async def main():
    """执行旧版向量缓存文件格式转换与指纹核验"""
    path = ROOT / 'data/local_index/embeddings.json'
    if not path.exists():
        return
    cache = json.loads(path.read_text(encoding='utf-8'))
    # 如果已经包含 entries 字典，说明已经是新版格式，无需重复迁移
    if 'entries' in cache:
        print("向量缓存已经是新版格式，无需迁移。")
        return
        
    rows, _ = load_corpus()
    prefixes = ('admin-review-2023/', 'civil-code-2020/', 'civil-procedure-2023/', 'criminal-law-2023/')
    old = [dict(r) for r in rows if not r.get('field') or r['id'].startswith(prefixes)]
    old = [r for r in old if r['id'] != 'civil-procedure-2023/第三百零六条']
    for row in old:
        if row['id'] == 'criminal-law-2023/第四百五十二条':
            row['text'] = row['text'].splitlines()[0]
            
    model = OllamaClient()
    try:
        await model.status()
        expected = hashlib.sha256(json.dumps([model.model_id, [(r['id'], r['text']) for r in old]], ensure_ascii=False).encode()).hexdigest()
        if cache.get('fingerprint') != expected or len(old) != len(cache.get('vectors', [])):
            raise RuntimeError('旧向量指纹未匹配，拒绝复用')
            
        entries = {hashlib.sha256((r['id'] + '\n' + r['text']).encode()).hexdigest(): v for r, v in zip(old, cache['vectors'])}
        # 备份原文件
        path.with_suffix('.v1.backup.json').write_bytes(path.read_bytes())
        tmp = path.with_suffix('.migration.tmp')
        tmp.write_text(json.dumps(dict(model_id=model.model_id, entries=entries)), encoding='utf-8')
        tmp.replace(path)
        print('CACHE_MIGRATED_FINGERPRINT_CONFIRMED', len(entries))
    finally:
        await model.close()


if __name__ == '__main__':
    asyncio.run(main())

