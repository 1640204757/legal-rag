"""真实端到端评测命令：python scripts/run_rag_eval.py。

输入为已固化retrieval_eval_v2.json和本机现有Qdrant/Ollama；输出真实JSON、Markdown及逐题网络证据。
不改题、不训练、不写Qdrant、不更改生产拦截器；顺序执行40题，失败仍保留，不自动重试挑分。
"""
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_rag.eval_runtime import run_evaluation


def main():
    """使用唯一运行目录保存证据；同名总报告存在时拒绝覆盖，保留首次实测结果。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/local-rag/eval_report_20261021.json')
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise SystemExit('报告已存在，请明确指定新的--output，不能覆盖首次实测结果')
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    directory=ROOT/'artifacts/local-rag/eval-runs'/stamp
    report=asyncio.run(run_evaluation(args.output,directory))
    return 0 if report['summary']['verdict']=='合格' else 2


if __name__=='__main__':
    sys.exit(main())
