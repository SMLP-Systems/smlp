import sys
from bnh_z3_minimax import solve_bnh_z3_minimax
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_z3_minimax(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_z3'])
        result = solve_bnh_z3_minimax(test_path)
        assert result == "62d56d7634ede2a15b0670628fabce38aed2c35c3ced10ecac7dfdef6d393c76" or \
               result == "be053b234db4ef16fa872dc48f32ceb2ce26a990b4f011d6d063fe1dbb55f2f2" or \
               result == "13d26e73589326b1626c27f6fc44479e5a1600d01ebb1a95fbe714c88a715c7c"
