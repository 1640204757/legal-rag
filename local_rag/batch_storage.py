"""批次文件的保守发布与恢复：保留旧记录，完整目录再进入可加载资料区。

入口update_reviewed_record用于补证，publish_document用于新增正文。
函数会写文件或移动完整暂存目录，不联网、不改数据库、不删除旧证据。
调用者仍须排他执行并预先核对哈希及正文，这里只负责存储边界。
"""
import json
from pathlib import Path

from .catalog_ingest import _save_json


def update_reviewed_record(path, before, new, backup):
    """原子更新一份记录，允许中断后重放已完成项；未知当前状态拒绝覆盖。"""
    current = path.read_bytes()
    if current != before and json.loads(current) != new:
        raise ValueError('当前记录既不是原状态也不是本批目标状态')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.read_bytes() != before:
            raise ValueError('既有备份与本批原状态不一致')
    else:
        with backup.open('xb') as stream:
            stream.write(before)
    if json.loads(current) != new:
        _save_json(path, new)


def publish_document(target, stage, files):
    """在资料库外补齐暂存文件后同盘改名发布；已发布内容一致时只核验。

    target与stage须在同一文件系统，stage由本批独占。若目标有未知内容则保留并报错。
    暂存过程被打断不会产生可加载的半份record；重跑只补相同内容的缺失文件。
    """
    if any(Path(name).name != name or name in {'', '.', '..'} for name in files):
        raise ValueError('只接受本目录文件名')
    if target.exists():
        if set(p.name for p in target.iterdir()) != set(files) or any(
            not (target / name).is_file() or (target / name).read_bytes() != value
            for name, value in files.items()
        ):
            raise ValueError('目标目录已存在且内容不同，不覆盖')
        return
    stage.mkdir(parents=True, exist_ok=True)
    if any(p.name not in files for p in stage.iterdir()):
        raise ValueError('暂存目录包含未知文件')
    for name, value in files.items():
        path = stage / name
        if path.exists() and path.read_bytes() == value:
            continue
        # 暂存文件可能在上次断电时只写了一部分，允许重写本批暂存产物。
        path.write_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
