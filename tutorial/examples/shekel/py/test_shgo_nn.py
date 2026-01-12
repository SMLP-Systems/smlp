#!/usr/bin/python3.12
import sys
from shgo_nn import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
import logging

def test_shgo_nn(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        logging.disable(logging.CRITICAL)
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['shgo_nn'])
        assert main(test_path) == '9b121952e6fc76160ac694b8bb45f292a07b2485aadfc7d653a75cfedca08cc2'
