"""会话冷归档与远期原话回查（多轮会话底座第 2 步）。

职责：
1. 把每个会话逐轮追加到 ``data/sessions/{sid}.jsonl``，一个事件一行；
   写入通过 ``asyncio.to_thread`` 落到线程，不阻塞事件循环。
2. 读取时逐行解析，自动跳过末尾半截行或损坏行（掉电/进程被杀后的常见残片）。
3. 只在**用户原话**上做 BM25 回查，绝不把助手的推测或格式套话当作用户事实。
4. 提供 ``delete_session_archive`` 供用户主动清空本地记录。

边界：
- 归档是审计与回查材料，不是法律证据；回查命中也不证明事实或法律适用。
- 归档失败不能影响问答主流程（调用方 service.py 会吞掉异常并继续）。
- 本模块只读写本机 ``data/sessions``，不联网、不写数据库。
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path

from .retrieval import BM25Index, tokens

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR_ENV = "LEGAL_RAG_SESSION_DIR"
SID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# 回查触发：明显指代或“第X轮”引用；没有这些词时要靠高词面重合兜底。
REFERENCE_CUES = (
    "前面说的", "前面说", "前面提到", "前面提", "前面那",
    "之前说的", "之前说", "之前提到", "之前提", "之前那",
    "刚才说的", "刚才说", "刚才提到", "刚才提",
    "上次说的", "上次说", "上次提到", "上次提",
    "一开始", "最开始", "最早说", "当时说",
    "借条", "那笔", "这笔",
)
ROUND_REFERENCE = re.compile(r"第[零〇一二三四五六七八九十百千万两\d]+轮")

# 没有指代词时，必须和某条早期原话有足够多共同词项才注入，避免把无关旧话塞进提示词。
MIN_SHARED_TOKENS = 6
MIN_OVERLAP_RATIO = 0.5

# 单片（含固定前缀与引号）不超过 200 字节，全部回查片段合计不超过 500 字节。
RECALL_PIECE_BYTES = 200
RECALL_TOTAL_BYTES = 500
RECALL_LIMIT = 2
# 最近一轮已经由历史视图承担，回查只找更早的原话，减少重复占字节。
DEFAULT_EXCLUDE_RECENT = 1


def archive_dir(directory=None):
    """返回归档目录：显式参数 > 环境变量 > 项目 data/sessions；只计算路径，不创建目录。"""
    if directory is not None:
        return Path(directory)
    override = os.environ.get(ARCHIVE_DIR_ENV)
    return Path(override) if override else ROOT / "data" / "sessions"


def archive_path(sid, directory=None):
    """返回会话文件路径；会话编号只允许安全字符，避免路径穿越。"""
    if not isinstance(sid, str) or not SID_PATTERN.fullmatch(sid):
        raise ValueError("会话编号格式不合法")
    return archive_dir(directory) / f"{sid}.jsonl"


def append_event_sync(sid, event, directory=None):
    """同步追加一个事件；每条一行 JSON，flush + fsync 后才返回。"""
    path = archive_path(sid, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


async def append_event(sid, event, directory=None):
    """异步追加事件：在线程中落盘，避免磁盘写入卡住事件循环。"""
    return await asyncio.to_thread(append_event_sync, sid, event, directory)


def read_events(sid, directory=None):
    """按顺序读取会话事件；跳过空行、损坏行和末尾半截行，不修改文件。"""
    path = archive_path(sid, directory)
    if not path.exists():
        return []
    events = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # 掉电或进程被杀时最后一行可能只写了一半；跳过即可，不能让整份档案失效。
                    continue
                if isinstance(event, dict) and event.get("role") in ("user", "assistant"):
                    events.append(event)
    except OSError:
        return []
    return events


def read_user_turns(sid, directory=None):
    """只取用户原话并附上轮次，供回查使用；助手回答在这里被彻底排除。"""
    turns, fallback_round = [], 0
    for event in read_events(sid, directory):
        if event.get("role") != "user":
            continue
        content = event.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        fallback_round += 1
        round_no = event.get("round")
        if not isinstance(round_no, int) or round_no < 1:
            round_no = fallback_round
        turns.append({"round": round_no, "content": content})
    return turns


def turns_from_history(history):
    """归档不可用时从内存历史兜底：只取 user 消息，按出现顺序编号。"""
    turns = []
    for message in history or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            turns.append({"round": len(turns) + 1, "content": content})
    return turns


def delete_session_archive(sid, directory=None):
    """删除单个会话的本地档案；返回是否真的删掉了文件，缺失不算错误。"""
    path = archive_path(sid, directory)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _clip_bytes(text, limit):
    """按 UTF-8 字节截断，避免把汉字切半；不修改输入。"""
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text or ""
    return raw[:max(limit, 0)].decode("utf-8", "ignore")


def _overlap(question, content):
    """返回（共同词项数，占问句词项比例）；用于判断“是否高度相关”，不做法律语义分析。"""
    query_terms = set(tokens(question or ""))
    if not query_terms:
        return 0, 0.0
    shared = query_terms & set(tokens(content or ""))
    return len(shared), len(shared) / len(query_terms)


def _has_reference_cue(question):
    """是否出现明显指代或“第X轮”；只是触发信号，不代表一定命中。"""
    text = question or ""
    return bool(ROUND_REFERENCE.search(text)) or any(cue in text for cue in REFERENCE_CUES)


def format_recall_line(round_no, content, piece_bytes=RECALL_PIECE_BYTES):
    """按固定格式生成一条前情陈述，整行不超过 piece_bytes 字节；放不下正文时返回空串。"""
    prefix = f"【前情陈述（第{round_no}轮用户原话）】：“"
    suffix = "”"
    overhead = len(prefix.encode("utf-8")) + len(suffix.encode("utf-8"))
    budget = piece_bytes - overhead
    if budget <= 0:
        return ""
    normalized = " ".join((content or "").split())
    body = _clip_bytes(normalized, budget)
    if not body:
        return ""
    if len(body.encode("utf-8")) < len(normalized.encode("utf-8")) and budget >= 6:
        # 有截断时补省略号；先给省略号留出位置，连同它一起也必须守住单片字节上限。
        body = _clip_bytes(normalized, budget - len("……".encode("utf-8"))) + "……"
    return prefix + body + suffix


def recall(question, turns, *, limit=RECALL_LIMIT, exclude_rounds=None,
           current_round=None, exclude_recent=DEFAULT_EXCLUDE_RECENT,
           piece_bytes=RECALL_PIECE_BYTES, total_bytes=RECALL_TOTAL_BYTES):
    """在用户历史原话中做 BM25 定向回查，返回最多 limit 条可注入片段。

    turns 必须是 ``[{"round": int, "content": str}]`` 形式的**用户原话**列表，
    模块不会读取助手回答。返回每个命中包含 round/content/score/line；
    没有明显指代且词面重合不足时返回空列表，交给模型在“## 追问”里澄清。

    ``current_round`` 用于排除本轮与最近一轮：最近的话已经在历史视图里，
    回查只负责更早、可能已被预算裁掉的远期原话。
    """
    if not question or not turns:
        return []
    excluded = set(exclude_rounds or ())
    latest = current_round if current_round is not None else max(turn["round"] for turn in turns) + 1
    if exclude_recent:
        excluded.update(range(latest - exclude_recent, latest + 1))
    else:
        excluded.add(latest)
    candidates = [turn for turn in turns if turn["round"] not in excluded and turn["content"].strip()]
    if not candidates:
        return []

    index = BM25Index([{"text": turn["content"]} for turn in candidates])
    ranking = index.rank(question)
    cue = _has_reference_cue(question)
    hits, used_bytes = [], 0
    for position, score in ranking:
        turn = candidates[position]
        shared, ratio = _overlap(question, turn["content"])
        # 有指代词时只要求有共同词项；没有指代词时必须“高度相关”，否则不注入。
        if not cue and (shared < MIN_SHARED_TOKENS or ratio < MIN_OVERLAP_RATIO):
            continue
        line = format_recall_line(turn["round"], turn["content"], piece_bytes)
        if not line:
            continue
        line_bytes = len(line.encode("utf-8"))
        if used_bytes + line_bytes > total_bytes:
            continue
        hits.append({"round": turn["round"], "content": turn["content"],
                     "score": round(score, 4), "line": line})
        used_bytes += line_bytes
        if len(hits) >= limit:
            break
    return hits


def archive_user_event(round_no, content, timestamp=None):
    """构造用户提问事件；index 与 round 一致，便于按轮次回查。"""
    return {"role": "user", "content": content, "round": round_no, "index": round_no,
            "timestamp": timestamp if timestamp is not None else time.time()}


def archive_answer_event(round_no, *, raw_answer, answer="", blocked=False, blocking_issues=None,
                         materials_sent=None, status="done", timestamp=None):
    """构造回答事件；即使被拦截也保留原始输出、拦截原因和实际送模材料编号。"""
    return {
        "role": "assistant",
        "raw_answer": raw_answer or "",
        "answer": answer or "",
        "blocked": bool(blocked),
        "blocking_issues": list(blocking_issues or []),
        "materials_sent": list(materials_sent or []),
        "status": status,
        "round": round_no,
        "index": round_no,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }
