import sys
from bnh_csv_decision_tree_generated_constraints import main
from os import remove, popen
from os.path import exists, realpath, dirname
from sys import version
from os import getenv
import sys

def test_bnh_csv_decision_tree_generated_constraints(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        sys.path.append('test_path')
        print("")
        m.setattr(sys, 'argv', ['bnh_csv_decision_tree_generated_constraints'])
        if version.split()[0] == '3.14.2':
            assert main(test_path) == "8a17cb976a0b670c6b874201065dfca9bec12dc5eb879b49dc2d356ceb4c4f0f"
        else:
            assert main(test_path) == "394e9980b014c68cf201eedbbe85ef74251a7f3ad4d9ad7ce0f1db86b85cb0a0"
