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
        assert main(test_path,5000) == "ea1d8a61eb9e10579addbdbab7dc0c8f65a6446f7273d3c68f42c9f76749b287"
