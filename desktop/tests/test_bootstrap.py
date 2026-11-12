"""The entry point must report dependency failures before the launcher imports."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["ImportError", "OSError", "RuntimeError"])
def test_dependency_import_failure_shows_sanitized_native_error(failure: str) -> None:
    script = """
import builtins
import ctypes
import json
import runpy
import sys

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise getattr(builtins, sys.argv[1])('secret-token C:/private/model.pth')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
ctypes.windll.user32.MessageBoxW = lambda *args: print(json.dumps(args, ensure_ascii=False))
runpy.run_module('desktop', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, failure],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 1
    assert result.stderr == ""
    dialog = json.loads(result.stdout)
    assert dialog[0] is None
    assert "cài lại" in dialog[1]
    assert "Tomato" in dialog[2]
    assert dialog[3] == 0x10
    assert "secret-token" not in result.stdout
    assert "C:/private" not in result.stdout


def test_bootstrap_preserves_launcher_exit_status() -> None:
    script = """
import runpy
import sys
from types import ModuleType
launcher = ModuleType('desktop.launcher')
launcher.main = lambda: 7
sys.modules['desktop.launcher'] = launcher
runpy.run_module('desktop', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7
    assert result.stdout == result.stderr == ""


def test_native_shutdown_gc_skips_imports_but_collects_session_objects() -> None:
    script = """
import atexit, gc, runpy, sys, weakref
from types import ModuleType

class Node:
    pass

imported = Node()
imported.cycle = imported
def tracked():
    return any(item is imported for item in gc.get_objects())
atexit.register(lambda: print('interpreter cleanup:', tracked()))

def start():
    def native_cleanup():
        print('native cleanup:', tracked())
        session = Node()
        session.cycle = session
        reference = weakref.ref(session)
        del session
        gc.collect()
        print('session reclaimed:', reference() is None)
    atexit.register(native_cleanup)
    return 0

launcher = ModuleType('desktop.launcher')
launcher.main = start
sys.modules['desktop.launcher'] = launcher
runpy.run_module('desktop', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.splitlines() == [
        "native cleanup: False",
        "session reclaimed: True",
        "interpreter cleanup: True",
    ]
