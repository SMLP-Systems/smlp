import sys
from bnh_csv import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_bnh_csv(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_csv'])
        if version.split()[0] == '3.14.2':
            assert main(test_path) == "8e5f3eeefd9919cf27e458765f3b5a58661d21d83a49b47cbdc802a0d4bf17e8"
        else:
            assert main(test_path) == "9fdec865e276c939aa69b5f677702e3ebf3aae4919d1bd374d2e84c49f17ddd7"
