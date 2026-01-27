import sys
from constraint_dora_pysmt_z3 import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv

def test_constraint_dora(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['constraint_dora_z3'])
        assert main(test_path,10000) == '28e625390e575991d2d74547f6cbd05cf72c7e53c6f1b5ebcfa19ba14e63d4fb'
