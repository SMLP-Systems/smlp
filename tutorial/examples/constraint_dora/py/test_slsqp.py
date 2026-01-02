import sys
from constraint_dora import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv

def test_constraint_dora(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out =  test_path + '/dataset.txt'
        if exists(out):
            remove(out)
        print("")
        assert exists(out) == False
        m.setattr(sys, 'argv', ['constraint_dora'])
        assert main(1000, test_path) == 'a757d8ff6f40eacdbc7de3dd6c8afae2041f9bc2c7cd706ee0f0660f59d33803'
        assert int(popen(f"sum {out}").read().split()[0]) == 61752
