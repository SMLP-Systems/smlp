import sys
from bnh_csv_sklearn import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version_info
from os import getenv

def test_bnh_csv_sklearn(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_csv_sklearn',test_path,5000,True])
        if version_info.minor == 14:
            assert main(test_path) == "b14ee70fb066bc7c627780b5feb389f8de0740640db36dbf5a5b183dd99e9b75"
        else:
            assert main(test_path) == "24f9d889fc93c6789d55b6f63843fb0ae0fd24170b56bd41f95c21419d7aa576"
