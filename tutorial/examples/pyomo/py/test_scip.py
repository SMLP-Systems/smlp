import sys
from scip import main
from os import remove, popen
from os.path import realpath, dirname
from sys import version
from os import getenv

def test_scipy(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['scipy'])
        assert main(test_path,5000) == "63af7a9f330083a8164db242198801fd756f2e94e19485ec245144b000f69a99"
