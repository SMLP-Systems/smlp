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
        assert main(test_path) == "0bba1875c7fa2f11bc2118e234372fa5ad8ec2c1e8de0692b23358d45624a3c9"
