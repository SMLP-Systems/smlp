#!/usr/bin/python3.12
import sys
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
from pytest import mark
import logging

@mark.skipif(sys.version_info >= (3, 14), 
                   reason="Skipping Keras tests for Python 3.14")
def test_shgo_nn(monkeypatch, request):
    from shgo_nn import main
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        logging.disable(logging.CRITICAL)
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['shgo_nn'])
        assert main(test_path) == '9b121952e6fc76160ac694b8bb45f292a07b2485aadfc7d653a75cfedca08cc2'
