#!/usr/bin/python3.12
import sys
from c3dtlz4_ex import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_gpsampler(monkeypatch, request):
    root_dir = str(request.config.rootpath) + "/"
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out =  test_path + '/results.csv'
        if exists(out):
            remove(out)
        print("")
        assert exists(out) == False
        m.setattr(sys, 'argv', ['c3dtlz4_ex','-n','100','-p',test_path])
        assert main() == '5ad02bc5986c77fbe0fb2d088aaa6a55928dfda0c8b29d357ff6e407cd409b5b'
        if version.split()[0] == '3.14.2':
            assert int(popen(f"sum {out}").read().split()[0]) == 35272
        else:
            assert int(popen(f"sum {out}").read().split()[0]) == 47204
