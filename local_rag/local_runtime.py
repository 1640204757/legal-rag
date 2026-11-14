"""本机运行约定：目录身份和模型目录解析，供Web、启动器与模型适配器共同使用。

入口workspace_identity/models_directory；只计算路径或摘要，不读凭证、不创建目录、不搬移模型。
目录摘要用于避免误连另一份项目，不是认证令牌或进程终止凭据。
"""
import hashlib
import os
from pathlib import Path

APP_ID = 'openlawguide-local'


def workspace_identity(root):
    """返回当前项目目录的稳定摘要；Web状态不直接暴露本机绝对路径。"""
    value = os.path.normcase(str(Path(root).resolve()))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def models_directory(root, environ=None, *, home=None):
    """显式环境变量优先，其次既有项目/用户模型，新安装默认项目内；不创建或迁移目录。

    已运行的Ollama仍以其自身启动环境为准，客户端不能更改它。
    """
    environ = os.environ if environ is None else environ
    if environ.get('OLLAMA_MODELS'):
        return str(Path(environ['OLLAMA_MODELS']).expanduser().resolve())
    project = Path(root).resolve() / 'data/ollama_models'
    user = Path(home if home is not None else Path.home()) / '.ollama/models'
    marker = 'manifests/registry.ollama.ai/library/qwen3'
    for candidate in (project, user):
        if (candidate / marker).is_dir():
            return str(candidate)
    return str(project)
