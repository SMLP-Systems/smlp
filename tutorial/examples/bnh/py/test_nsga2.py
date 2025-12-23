#!/usr/bin/python3.12
import sys
from bnh import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_nsga2(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out  = test_path + '/NSGA2_pareto.txt'
        out1 = test_path + '/dataset.csv'
        for o in [out, out1]:
            if exists(out):
                remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['bnh'])
        assert main(test_path) == 0
        if version.split()[0] == '3.14.2':
            assert int(popen(f"sum {out}").read().split()[0]) == 19751
        else:
            assert int(popen(f"sum {out}").read().split()[0]) == 43431
        assert int(popen(f"sum {out1}").read().split()[0]) == 9816
