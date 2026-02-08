#!/usr/bin/python3.12
import sys
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
from pytest import mark

@mark.skipif(sys.version_info >= (3, 14), 
                   reason="Skipping Keras tests for Python 3.14")
def test_keras2onnx(monkeypatch, request):
    from keras2onnx import main
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        onnx_model =  test_path + '/shekel_model_expected.onnx'
        if exists(onnx_model):
            remove(onnx_model)
        assert exists(onnx_model) == False
        print("")
        m.setattr(sys, 'argv', ['shgo_onnx'])
        assert main(test_path) == 'e833c8d813f8a3467c8ce9fa72b3f3ce5be4f38f1a38ce99fdc0c362952cfe54'
