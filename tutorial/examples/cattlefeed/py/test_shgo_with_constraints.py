#!/usr/bin/python3.12
import sys
from optimization_ex1 import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv

def test_optimization_ex(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['optimization_ex1'])
        assert main() == '84ef32981634734011dcab92dc3b2609b4b6107a96dff947bfbc3c090dbaf2d4'
