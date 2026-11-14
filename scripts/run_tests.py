"""兼容旧测试入口：自动定位项目，测试临时文件仅放在项目内；推荐入口为run_workspace_tests.py。"""
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local-deps"))

# 临时目录随项目位置确定，不依赖作者本机的盘符或用户名。
basetemp = ROOT / ".tmp_test"
basetemp.mkdir(parents=True, exist_ok=True)

import pytest
if __name__ == "__main__":
    exit_code = pytest.main([
        str(ROOT / "tests_local"),
        str(ROOT / "prototype/tests"),
        "-q",
        "--basetemp", str(basetemp),
        "--rootdir", str(ROOT)
    ])
    print("pytest exit code:", exit_code)
    sys.exit(exit_code)
