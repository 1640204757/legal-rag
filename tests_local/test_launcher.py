"""启动安全：不得误停已有服务、不得相信陈旧PID，自检不启动服务且不输出令牌。"""
import pytest

import run_local as launch


def test_existing_ollama_with_other_models_is_never_killed(monkeypatch):
    commands=[]
    monkeypatch.setattr(launch,'available',lambda url:{'models':[{'name':'another-local-model'}]})
    monkeypatch.setattr(launch.shutil,'which',lambda name:'ollama.exe')
    monkeypatch.setattr(launch.subprocess,'run',lambda command,**kw:commands.append(command))
    monkeypatch.setattr(launch.subprocess,'Popen',lambda command,**kw:commands.append(command))
    monkeypatch.setattr(launch.time,'sleep',lambda delay:None)
    launch.ensure_ollama()
    assert commands==[], '已有模型服务不能因标签不同被杀掉或替换'


def test_legacy_pid_file_cannot_authorize_ending_any_process(monkeypatch,tmp_path):
    commands=[]
    runtime=tmp_path/'data/runtime';runtime.mkdir(parents=True)
    (runtime/'server.pid').write_text('123456',encoding='ascii')
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(launch,'available',lambda url:None)
    monkeypatch.setattr(launch.subprocess,'run',lambda command,**kw:commands.append(command))
    monkeypatch.setattr(launch,'port_in_use',lambda host,port:False)
    launch.existing_web_status()
    assert commands==[]
    assert (runtime/'server.pid').read_text(encoding='ascii')=='123456'


def test_another_workspace_on_the_same_port_is_left_running(monkeypatch):
    monkeypatch.setattr(launch,'available',lambda url:{'app_id':'openlawguide-local','workspace_id':'another-copy'})
    with pytest.raises(RuntimeError,match='目录|身份'):
        launch.existing_web_status()


def test_foreign_port_is_not_taken_over(monkeypatch):
    monkeypatch.setattr(launch,'available',lambda url:None)
    monkeypatch.setattr(launch,'port_in_use',lambda host,port:True)
    with pytest.raises(RuntimeError,match='端口'):
        launch.existing_web_status()


def test_reusing_our_web_never_starts_or_stops_processes(monkeypatch,capsys):
    from local_rag.local_runtime import workspace_identity
    status={'app_id':'openlawguide-local','workspace_id':workspace_identity(launch.ROOT),'token':'DO_NOT_PRINT_TOKEN',
            'chat_ready':True,'embedding_ready':True,'index':{'state':'ready'},'database':{'ready':True}}
    monkeypatch.setattr(launch,'available',lambda url:status)
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('复用时不得启动新进程'))
    monkeypatch.setattr(launch.subprocess,'run',lambda *a,**k:pytest.fail('复用时不得结束进程'))
    assert launch.main(['--once','--no-browser'])==0
    assert 'DO_NOT_PRINT_TOKEN' not in capsys.readouterr().out


def test_check_mode_does_not_start_services_or_print_tokens(monkeypatch,capsys):
    monkeypatch.setattr(launch,'check_environment',lambda:{'ready':False,'token':'DO_NOT_PRINT_TOKEN',
        'python_ok':True,'dependencies_ok':True,'data_ok':True,'ollama_connected':False,'models_ready':False,
        'qdrant_ready':False,'web_state':'free'})
    monkeypatch.setattr(launch,'ensure_ollama',lambda:pytest.fail('自检不能启动Ollama'))
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('自检不能启动Web'))
    assert launch.main(['--check'])==2
    assert 'DO_NOT_PRINT_TOKEN' not in capsys.readouterr().out


def test_only_the_owned_child_is_terminated():
    class Child:
        terminated=False
        def poll(self):return None
        def terminate(self):self.terminated=True
        def wait(self,timeout):return 0
    child=Child()
    launch.stop_owned_child(child)
    assert child.terminated


def test_explicit_model_directory_takes_precedence_without_creation(tmp_path):
    from local_rag.local_runtime import models_directory
    installed=tmp_path/'data/ollama_models/manifests/registry.ollama.ai/library/qwen3'
    installed.mkdir(parents=True)
    configured=tmp_path/'chosen-models'
    assert models_directory(tmp_path,{'OLLAMA_MODELS':str(configured)},home=tmp_path/'home')==str(configured.resolve())
    assert not configured.exists(), '目录解析不能创建文件或搬移模型'


@pytest.mark.parametrize('value',['https://127.0.0.1:11434','http://example.org:11434',
                                'http://user:secret@localhost:11434','http://localhost:11434/api'])
def test_service_configuration_stays_on_loopback(monkeypatch,value):
    monkeypatch.setenv('OLLAMA_LOCAL_URL',value)
    with pytest.raises(RuntimeError,match='本机HTTP'):
        launch.service_url('OLLAMA_LOCAL_URL','http://127.0.0.1:11434')


@pytest.mark.parametrize('status',[['unexpected'],{'app_id':'openlawguide-local','workspace_id':'different','process_id':42}])
def test_wait_rejects_foreign_or_malformed_status(monkeypatch,status):
    class Child:
        pid=42
        def poll(self):return None
    monkeypatch.setattr(launch,'available',lambda url:status)
    with pytest.raises(RuntimeError,match='不是本次'):
        launch.wait_for_web(Child(),timeout=1)


def test_python_config_accepts_bom_and_rejects_missing_path(monkeypatch,tmp_path):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.delenv('LOCAL_RAG_PYTHON',raising=False)
    python=tmp_path/'python.exe';python.touch()
    config=tmp_path/'.local-python-path.txt'
    config.write_text(str(python),encoding='utf-8-sig')
    assert launch.selected_python()==str(python.resolve())
    config.write_text(str(tmp_path/'missing.exe'),encoding='utf-8')
    with pytest.raises(RuntimeError,match='Python路径无效'):
        launch.selected_python()
