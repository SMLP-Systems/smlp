import sys
from bnh_pysmt_z3 import solve_bnh_pysmt
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_z3_pysmt(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_pysmt_z3'])
        assert solve_bnh_pysmt(test_path) == "e12e88904ce51e361c48c48a201201779d0049ffa7fa88353cbbd74c90c565a5"
