import sys
from s2_tx_anonym import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv

def test_s2_tx_anonym_csv(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['s2_tx_anonym', '-p', test_path, '-d','17', '-n', '18', '-s', '20'])
        assert main() == "a5e34617258a42f2fcd5bb8501bfffc4a75ce835a5805aefbed2d3640672fbb9"
