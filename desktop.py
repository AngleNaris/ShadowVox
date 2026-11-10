"""桌面程序及同一 EXE 的离线验收入口。"""
import sys
from pathlib import Path

if not getattr(sys,"frozen",False):
    sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))


def main():
    if len(sys.argv)==3 and sys.argv[1] in ('--density-worker','--master-worker'):
        from shadowvox.vst_worker import serve
        if sys.argv[1] == '--density-worker':
            from shadowvox.density import densify as render
        else:
            from shadowvox.mastering import finalize as render
        return serve(Path(sys.argv[2]), render)
    # 冷启动离线验收：拒绝 Python 层外部连接；正常运行不启用该钩子。
    import os
    if os.environ.get('SHADOWVOX_OFFLINE_TEST') == '1':
        import socket
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        def local_only(address):
            return isinstance(address, tuple) and (address[0] in ('localhost','::1') or str(address[0]).startswith('127.'))
        def connect(sock,address):
            if not local_only(address): raise RuntimeError('离线验收禁止外部连接')
            return original_connect(sock,address)
        def connect_ex(sock,address):
            if not local_only(address): raise RuntimeError('离线验收禁止外部连接')
            return original_connect_ex(sock,address)
        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
    if len(sys.argv)==5 and sys.argv[1]=="--plugin-editor":
        try:
            from shadowvox.vst_host import open_editor
            open_editor(sys.argv[2],sys.argv[3])
        except Exception:
            import traceback
            Path(sys.argv[4]).write_text(traceback.format_exc(),encoding="utf-8")
            return 1
        return 0
    if len(sys.argv)>2 and sys.argv[1]=="--process":
        import json
        from shadowvox.pipeline import run,Options
        request=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        log_path=Path(request['output']).with_suffix('.log')
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('w',encoding='utf-8') as handle:
            def log(message):
                handle.write(message+'\n');handle.flush()
            try:
                run(request['dry'],request['song'],request['output'],work_dir=request.get('work'),
                    opts=Options(strength=request.get('strength',.75),pitch_mode='register',max_shift_cents=450,
                                 mix_effects=request.get('mix_effects',True),renderer='psola',rhythm_mode='full',
                                 master_density=request.get('master_density'),
                                 fixed_master=request.get('fixed_master',False),
                                 dry_has_backing=request.get('dry_has_backing',False),
                                 mute_reference_gaps=request.get('mute_reference_gaps',False),
                                 reuse_stems=request.get('reuse_stems',True)),log=log)
            except Exception:
                import traceback
                log(traceback.format_exc());return 1
        return 0
    from shadowvox.gui import main as gui_main
    return gui_main(sys.argv[2] if len(sys.argv)>2 and sys.argv[1]=='--gui-check' else None)


if __name__=='__main__':
    raise SystemExit(main())
