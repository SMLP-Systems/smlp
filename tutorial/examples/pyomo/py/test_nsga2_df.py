#!/usr/bin/python3.12
import sys
from nsga2 import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_nsga2(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out  = test_path + '/products_nsga2.csv'
        out1 = test_path + '/requirements_nsga2.csv'
        out2 = test_path + '/resources_nsga2.csv'
        for o in [out, out1]:
            if exists(out):
                remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['nsga2'])
        if version.split()[0] == '3.14.2':
            assert main(test_path) == "9278384310665b7de7e71662e0c4a7daba2e1323f512e86cddc8278e7f9bc37c"
        else:
            assert main(test_path) == "133c4021bb720a6d087a313a0f3138c255b5e2d9d152058a62f9e06cf16d4f23"
        assert int(popen(f"sum {out}").read().split()[0])  == 38360
        assert int(popen(f"sum {out1}").read().split()[0]) == 32435
        assert int(popen(f"sum {out2}").read().split()[0]) == 33263
