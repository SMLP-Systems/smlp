import sys
from bnh_csv_decision_tree import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version_info
from os import getenv

def test_bnh_csv_decision_tree(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['bnh_csv_decision_tree'])
        if version_info.minor == 14:
            assert main(test_path) == "8a17cb976a0b670c6b874201065dfca9bec12dc5eb879b49dc2d356ceb4c4f0f"
        else:
            assert main(test_path) == "394e9980b014c68cf201eedbbe85ef74251a7f3ad4d9ad7ce0f1db86b85cb0a0"
