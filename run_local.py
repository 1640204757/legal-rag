"""安全启动入口：只复用同目录Web，保留已有模型服务，仅管理本次创建的Web子进程。

阅读main→check_environment/ensure_ollama→wait_for_web；--check只读自检，--no-browser不打开浏览器。
不信任PID文件作停止凭据，不按进程名或端口结束进程。使用手册：docs/guides/USER_GUIDE.md。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.local-deps'))
from local_rag.local_runtime import APP_ID, models_directory, workspace_identity

LOCAL = 'http://127.0.0.1:18765'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """本机探测不跟随重定向，避免认证头或请求被本地错误配置导向远端。"""
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
_REPORT_FIELDS = ('ready','python_ok','python_version','dependencies_ok','missing_dependencies',
                  'data_ok','data_issue','coverage','ollama_connected','models_ready','missing_models',
                  'qdrant_ready','web_state')


def available(url):
    """只读本机JSON端点，禁用环境代理；不可达或非JSON返回None，不输出响应正文。"""
    try:
        with OPENER.open(url, timeout=2) as response:
            return json.load(response)
    except Exception:
        return None


def service_url(name, default):
    """读取本机基础地址；拒绝远端、凭证、路径和查询串，不把配置秘密拼入报错。"""
    value = os.environ.get(name, default).rstrip('/')
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                or port is not None and not 1 <= port <= 65535):
            raise ValueError()
    except ValueError:
        raise RuntimeError(f'{name}必须是无凭证的本机HTTP基础地址') from None
    return value


def port_in_use(host, port):
    """只探测TCP监听，不枚举PID、不停止或接管监听者。"""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def resolve_ollama_models_dir():
    """复用统一目录规则，不写盘或搬移已有模型。"""
    return models_directory(ROOT)


def existing_web_status():
    """仅复用相同目录Web；其他服务或旧版缺身份时保留运行并报错。"""
    status = available(LOCAL + '/api/status')
    if status is not None:
        if (not isinstance(status, dict) or status.get('app_id') != APP_ID
                or status.get('workspace_id') != workspace_identity(ROOT)):
            raise RuntimeError('18765端口已有服务，但无法确认属于当前项目目录；请在原启动窗口确认并停止，未自动结束进程')
        return status
    if port_in_use('127.0.0.1', 18765):
        raise RuntimeError('18765端口已占用或服务暂未响应；请检查原窗口/等待加载，未自动结束进程')
    return None


def selected_python():
    """选择环境变量、本机配置或当前解释器；错误配置报错，不静默改用其他Python。"""
    configured = ROOT / '.local-python-path.txt'
    candidate = os.environ.get('LOCAL_RAG_PYTHON')
    if not candidate and configured.is_file():
        candidate = configured.read_text(encoding='utf-8-sig').strip()
    path = Path(candidate or sys.executable).expanduser()
    if not path.is_file():
        raise RuntimeError('项目Python路径无效，请设置LOCAL_RAG_PYTHON或.local-python-path.txt')
    return str(path.resolve())


def child_environment():
    """只设置本次子进程的依赖路径与模型目录，不改系统环境或其他服务配置。"""
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'.local-deps')]),
                PYTHONIOENCODING='utf-8', OLLAMA_MODELS=resolve_ollama_models_dir())


def python_and_data_check():
    """用选定解释器只读校验依赖与语料；不建索引、不调用模型、不写库。"""
    code = '''import sys,json,importlib
result={'python_version':'.'.join(map(str,sys.version_info[:3])),'python_ok':sys.version_info[:2]==(3,12) and sys.maxsize>2**32,'dependencies_ok':False,'missing_dependencies':[],'data_ok':False}
if result['python_ok']:
    for name in ('fastapi','uvicorn','httpx','jsonschema','pytest'):
        try: importlib.import_module(name)
        except Exception: result['missing_dependencies'].append(name)
    result['dependencies_ok']=not result['missing_dependencies']
    if result['dependencies_ok']:
        try:
            from local_rag.retrieval import load_corpus
            rows,coverage=load_corpus()
            result['coverage']={k:coverage[k] for k in ('laws','cases','chunks')}
            result['data_ok']=bool(rows)
        except Exception as exc: result['data_issue']=type(exc).__name__
print(json.dumps(result,ensure_ascii=False))'''
    try:
        result = subprocess.run([selected_python(),'-B','-X','utf8','-c',code],cwd=ROOT,env=child_environment(),
                                capture_output=True,text=True,encoding='utf-8',timeout=60,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode != 0:
            raise RuntimeError('解释器自检失败；请使用64位CPython3.12并安装requirements-local.txt')
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError,subprocess.TimeoutExpired,ValueError,IndexError):
        raise RuntimeError('解释器/资料自检失败或超时，请核对Python、依赖与数据包；未启动服务') from None


def qdrant_available():
    """只读配置的本机Qdrant健康端点，不创建集合或读写资料。"""
    headers = {'api-key':os.environ['QDRANT_API_KEY']} if os.environ.get('QDRANT_API_KEY') else {}
    try:
        request=urllib.request.Request(service_url('QDRANT_LOCAL_URL','http://127.0.0.1:16333')+'/healthz',headers=headers)
        with OPENER.open(request,timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def check_environment():
    """只读汇总Python、依赖、数据和本机服务，不启动服务、不保存配置。"""
    report = python_and_data_check()
    tags = available(service_url('OLLAMA_LOCAL_URL','http://127.0.0.1:11434')+'/api/tags')
    names={m.get('name') for m in (tags.get('models',[]) if isinstance(tags,dict) else []) if isinstance(m,dict)}
    required=[os.environ.get('LOCAL_CHAT_MODEL','qwen3:4b-instruct-2507-q4_K_M'),
              os.environ.get('LOCAL_EMBED_MODEL','qwen3-embedding:0.6b')]
    report.update(ollama_connected=isinstance(tags,dict) and isinstance(tags.get('models'),list),
                  missing_models=[name for name in required if name not in names])
    report['models_ready']=not report['missing_models']
    report['qdrant_ready']=qdrant_available()
    try:
        report['web_state']='same_workspace' if existing_web_status() else 'free'
    except RuntimeError:
        report['web_state']='occupied_or_unverified'
    report['ready']=all(report.get(k,False) for k in ('python_ok','dependencies_ok','data_ok','ollama_connected','models_ready','qdrant_ready')) and report['web_state']!='occupied_or_unverified'
    return report


def ensure_ollama():
    """已有Ollama即使缺模型也只复用；仅端口空闲时启动，绝不结束已有模型进程。"""
    base=service_url('OLLAMA_LOCAL_URL','http://127.0.0.1:11434')
    tags=available(base+'/api/tags')
    if isinstance(tags,dict) and isinstance(tags.get('models'),list):
        return None
    parsed=urlsplit(base)
    if port_in_use(parsed.hostname,parsed.port or 80):
        raise RuntimeError('Ollama端口已占用但接口不可用；请检查已有服务，未自动停止它')
    ollama=shutil.which('ollama')
    if not ollama:
        candidate=Path(os.environ.get('LOCALAPPDATA',''))/'Programs/Ollama/ollama.exe'
        ollama=str(candidate) if candidate.is_file() else None
    if not ollama:
        raise RuntimeError('未找到Ollama，请先按使用手册安装；没有转用云端')
    env=dict(child_environment(),OLLAMA_NO_CLOUD='1',OLLAMA_HOST=parsed.netloc)
    child=subprocess.Popen([ollama,'serve'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for _ in range(40):
        tags=available(base+'/api/tags')
        if isinstance(tags,dict) and isinstance(tags.get('models'),list):
            return child
        if child.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError('Ollama未能启动，请手动核对服务与模型目录；未结束任何已有服务')


def stop_owned_child(child):
    """只停止本次Popen持有的Web进程句柄，不从PID文件、进程名或端口推断目标。"""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def wait_for_web(child, timeout=180):
    """等待本次子进程的HTTP身份，不把其他监听者当成功；大缓存加载可能较慢。"""
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if child.poll() is not None:
            break
        status=available(LOCAL+'/api/status')
        if status:
            if (not isinstance(status,dict) or status.get('app_id')!=APP_ID or status.get('workspace_id')!=workspace_identity(ROOT)
                    or status.get('process_id')!=child.pid):
                raise RuntimeError('端口响应不是本次项目实例；仅停止本次子进程，保留其他服务')
            return status
        time.sleep(0.5)
    raise RuntimeError('Web未能就绪，请查看data/runtime/server.log；仅停止本次Web子进程')


def announce(status, *, reused=False, browser=True):
    """仅打印白名单状态并按需打开本机页面，不输出/api/status中的令牌。"""
    print(('已复用当前目录的既有服务，未重新加载代码：' if reused else '本机Web已启动：')+LOCAL,flush=True)
    print(f"问答模型：{'就绪' if status.get('chat_ready') else '未就绪，请检查模型安装'}；"
          f"向量模型：{'就绪' if status.get('embedding_ready') else '未就绪'}；"
          f"索引：{status.get('index',{}).get('state','未知')}；"
          f"数据库：{'就绪' if status.get('database',{}).get('ready') else '未就绪/加载中'}",flush=True)
    if browser:
        try:webbrowser.open(LOCAL)
        except Exception:print('请手动在浏览器打开以上本机地址。',flush=True)


def main(argv=None):
    """启动或复用同目录Web；关闭窗口仅管理本次子进程，共享Ollama/Qdrant继续运行。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='只读自检，不启动/停止服务')
    parser.add_argument('--once',action='store_true',help='启动后退出启动器，Web留在后台；日常建议保持窗口')
    parser.add_argument('--no-browser',action='store_true',help='不自动打开浏览器')
    args=parser.parse_args(argv)
    if args.check:
        report=check_environment()
        print(json.dumps({k:report[k] for k in _REPORT_FIELDS if k in report},ensure_ascii=False,indent=2))
        return 0 if report.get('ready') else 2
    existing=existing_web_status()
    if existing:
        announce(existing,reused=True,browser=not args.no_browser)
        return 0
    check=python_and_data_check()
    if not all(check.get(k) for k in ('python_ok','dependencies_ok','data_ok')):
        raise RuntimeError('启动前自检未通过，请运行run_local.py --check并补齐Python、依赖和数据包')
    ollama_child=ensure_ollama()  # 共享模型服务不随Web窗口关闭，保留句柄直到启动器退出。
    runtime=ROOT/'data/runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    command=[selected_python(),'-B','-X','utf8','-m','uvicorn','local_rag.app:create_app','--factory',
             '--host','127.0.0.1','--port','18765','--no-access-log']
    with (runtime/'server.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=child_environment(),stdout=log,stderr=log,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        print('正在等待Web接口；首次资料/向量缓存加载可能较慢，请勿重复启动或重建。',flush=True)
        status=wait_for_web(child)
        (runtime/'server.pid').write_text(str(child.pid),encoding='ascii')  # 仅诊断，不作停止凭据。
        announce(status,browser=not args.no_browser)
        if args.once:return 0
        print('保持本窗口；Ctrl+C只停止本次Web，已有Ollama和Qdrant不受影响。',flush=True)
        while child.poll() is None:time.sleep(1)
        return child.returncode or 0
    except KeyboardInterrupt:
        stop_owned_child(child)
        return 0
    except Exception:
        stop_owned_child(child)
        raise


if __name__=='__main__':
    try:raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)
