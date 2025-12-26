import sys
from bnh_models_comparison import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_bnh_models_comparison(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_models_comparison'])
        assert main(test_path) == "4becad4a65fc95857ccaab85a29bd5571ec96c4455fb18bc7ef753f92f260ab8"
