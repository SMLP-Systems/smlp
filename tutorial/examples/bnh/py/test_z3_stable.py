import sys
from bnh_z3_stable import solve_bnh_z3
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_z3_stable(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_z3'])
        result = solve_bnh_z3(test_path)
        assert result == "f6837c61792b36166b0394286ad5da8e89e334b392e3aec4ffa9e02fa40f0e70" or \
               result == "08af985a9d077117f4bdc8240c0d4bdeec24265796b51162e4edf526b324c03d" or \
               result == "b101db1cba193f3dfbfcc5282fbc23a5a55d19e67916ab2f83d6fdbbccbea394"
