import sys
from bnh_pysmt_z3 import solve_bnh_pysmt
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_z3_bnh_pysmt(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_pysmt_z3'])
        assert solve_bnh_pysmt(test_path) == "2e0604a6b2b49fde902830f0376bcdd60d41ecf9551285adf3b544c5ca0874ad"
