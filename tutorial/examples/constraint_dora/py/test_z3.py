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
        assert main(test_path,10000) == '9405a560dd32fe3788a01b79963081072285d51063c798dd3585bfa4141ff489'
