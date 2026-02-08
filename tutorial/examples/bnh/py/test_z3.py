import sys
from bnh_z3 import solve_bnh_z3
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_z3(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_z3'])
        assert solve_bnh_z3(test_path) == "5a751f91bdaff952b945bb91951648c1b52c7fb937b61069cdeb2c85b8444e0f"
