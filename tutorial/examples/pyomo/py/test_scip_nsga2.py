import sys
from scip_nsga2 import main
from os import remove, popen
from os.path import realpath, dirname
from sys import version
from os import getenv

def test_scip_nsga2(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['scip_nsga2'])
        assert main(test_path,5000) == "49a5862ee69bac1b8630988dd5f0705233bb41b4d571c5a84376e6ec366e2e31"
