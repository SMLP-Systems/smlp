#!/usr/bin/python3.12
import sys
from pyomo_ex import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_pyomo(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out  = test_path + '/products.csv'
        out1 = test_path + '/requirements.csv'
        out2 = test_path + '/resources.csv'
        for o in [out, out1]:
            if exists(out):
                remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['pyomo_ex'])
        assert main(test_path) == "0caaeea85e5d176b9f6b9f4b8b364e3c94c406c0a24e9b1a8f384fe5a475edcc"
        assert int(popen(f"sum {out}").read().split()[0])  == 21277
        assert int(popen(f"sum {out1}").read().split()[0]) == 32435
        assert int(popen(f"sum {out2}").read().split()[0]) == 33263
