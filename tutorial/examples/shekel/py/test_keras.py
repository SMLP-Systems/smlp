#!/usr/bin/python3.12
import sys
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
from pytest import mark

@mark.skipif(sys.version_info >= (3, 14), 
        reason="Skipping Keras tests for Python 3.14")
def test_keras(monkeypatch, request):
    from keras_ex import main
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['keras_ex'])
        assert main(512, test_path) == '3be173406872257840c43d3c0a15f1115a87dd1145eac802b893832501ea3b60'
