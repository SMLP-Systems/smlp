import sys
from nsga2_mixed import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_nsga2_mixed(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['nsga2_mixed'])
        if version.split()[0] == '3.14.2':
            assert main(test_path) == "acfd6b4d8b5bc203779b89cd6d663ee37347936d3a7d6626750ce04818d9d906"
        else:
            assert main(test_path) == "3dcc4ef3a5fea092570ec4ec29249bd3f73849aac99294a9c0923b25e1381ee7"
