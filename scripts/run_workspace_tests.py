"""受限Windows工作区的pytest入口；隔离tmp_path，不写入历史artifacts目录。

仅在默认pytest临时目录被沙箱ACL拒绝时使用。项目测试断言照常执行；该适配只替换
tmp_path临时目录位置，并关闭pytest退出时对死符号链接的全路径扫描。
"""
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local-deps"))


def main(argv=None):
    """运行指定测试目录；默认运行本地测试和原型测试。"""
    import _pytest.tmpdir
    import pytest

    targets = argv if argv is not None else sys.argv[1:]
    targets = targets or ["tests_local", "prototype/tests"]
    resolved = []
    for value in targets:
        path = (ROOT / value).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            raise ValueError(f"测试路径不存在或越界：{value}")
        resolved.append(path.relative_to(ROOT).as_posix())

    temp_root = ROOT / f".pytest-workspace-{uuid.uuid4().hex}"
    temp_root.mkdir()
    # 测试默认把会话冷归档写到本次临时目录，既不污染 data/sessions，也不依赖本机残留。
    os.environ.setdefault("LEGAL_RAG_SESSION_DIR", str(temp_root / "sessions"))
    # 默认tmp_path通过pytest的0700目录创建，在本沙箱中会被ACL拒绝；改用普通mkdir继承权限。
    class WorkspaceTempPlugin:
        @pytest.fixture
        def tmp_path(self, request):
            slug = re.sub(r"[^A-Za-z0-9_-]+", "-", request.node.name)[:40]
            path = temp_root / f"{slug}-{uuid.uuid4().hex}"
            path.mkdir()
            try:
                yield path
            finally:
                if path.parent == temp_root and path.exists():
                    shutil.rmtree(path)

    # 自定义tmp_path fixture不调用TempPathFactory；session结束时也不扫描整个仓库树。
    _pytest.tmpdir.TempPathFactory.getbasetemp = lambda self: ROOT
    _pytest.tmpdir.cleanup_dead_symlinks = lambda _root: None
    print(f"WORKSPACE_TEMP_ROOT={temp_root}", flush=True)
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", *resolved], plugins=[WorkspaceTempPlugin()])
    finally:
        # 仅空目录可移除；失败时保留现场，绝不递归删除未知测试产物。
        if temp_root.parent == ROOT and temp_root.name.startswith(".pytest-workspace-"):
            try:
                temp_root.rmdir()
            except OSError:
                print(f"TEMP_EVIDENCE_PRESERVED={temp_root}", flush=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
