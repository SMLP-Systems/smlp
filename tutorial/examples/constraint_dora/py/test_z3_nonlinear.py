import sys
from constraint_dora_z3 import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv

def test_constraint_dora(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['constraint_dora_z3'])
        assert main(test_path,10000) == '16fba6eda91f97f8533e146143bdff2fc1d658193925eda86a62c44c6c37294a'
