import sys
from optimization_ex import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
import logging

def test_optimization_ex(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        logging.disable(logging.CRITICAL)
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out =  test_path + '/dataset.txt'
        if exists(out):
            remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['optimization_ex'])
        assert main(512, test_path) == '54e5105d59a57fd2898e581ca6f1e3502d4cda22b371fa17a88420d6da862602'
        assert int(popen(f"sum {out}").read().split()[0]) == 35930
        logging.disable(logging.NOTSET)
