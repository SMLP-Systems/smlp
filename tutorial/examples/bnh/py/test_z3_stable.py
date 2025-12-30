import sys
from bnh_z3_stable import solve_bnh_z3
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_z3_stable(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_z3'])
        assert solve_bnh_z3(test_path) == "b28ff663398cde468954641b095de693d2d6e201681e1e8cd0bada61c6bedbbd"
