import sys
from s2_rx_anonym import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_s2_rx_anonym_csv(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['s2_rx_anonym', '-p', test_path, '-d','17', '-n', '18', '-s', '20'])
        assert main() == "2d4a4633efd531baa2bef25e17175a1c4c9b4baa66ff1548a7059609850fa1f8"
