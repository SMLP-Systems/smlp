import sys
from easimple_ex import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_easimple(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out  = test_path + '/products_ga.csv'
        out1 = test_path + '/requirements_ga.csv'
        out2 = test_path + '/resources_ga.csv'
        for o in [out, out1]:
            if exists(out):
                remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['easimple_ex'])
        assert main(test_path) == "b3898956f280959c285691b5d6d8a08bf7779ab573ac5bbb2d0f5ddc1ae19d58"
        assert int(popen(f"sum {out}").read().split()[0])  == 20585
        assert int(popen(f"sum {out1}").read().split()[0]) == 32435
        assert int(popen(f"sum {out2}").read().split()[0]) == 33263
