#!/usr/bin/python3.12
import sys
from transform_variables import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv

def test_variables_transformation(monkeypatch, request):
    root_dir = str(request.config.rootpath) + "/"
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out =  test_path + '/transformed_variables.csv'
        if exists(out):
            remove(out)
        print("")
        assert exists(out) == False
        m.setattr(sys, 'argv', ['transform_variables','-db','example_expected.db', '-r','results_expected.csv','-p',test_path])
        assert main() == 'fb0b28c56f335de4c6e1e089ebc144e2c82d0f2e57ef8c55e7f5e603b4eb717d'
        assert int(popen(f"sum {out}").read().split()[0]) == 28924
