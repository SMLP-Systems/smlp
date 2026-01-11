#!/usr/bin/python3.12
import sys
from optimization_ex2 import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
import logging

def test_optimization_ex2(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        logging.disable(logging.CRITICAL)
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out =  test_path + '/shekel_meshgrid_26.csv'
        if exists(out):
            remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['optimization_ex2'])
        assert main(512, test_path) == '725607e769b9f5f23c633973efc6e7acf1953cb42999e81c1bc52e40ef264651'
        assert int(popen(f"sum {out}").read().split()[0]) == 695
