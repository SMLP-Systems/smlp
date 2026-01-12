#!/usr/bin/python3.12
import sys
from keras_ex import main
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
import logging

def test_keras(monkeypatch, request):
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        logging.disable(logging.CRITICAL)
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['keras_ex'])
        assert main(512, test_path) == '4b793e96580f5e8f393d97f1eb4c0a879fff288882f9c3de951a9e69be631808'
