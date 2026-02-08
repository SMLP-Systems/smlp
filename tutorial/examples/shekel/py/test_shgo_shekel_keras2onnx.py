#!/usr/bin/python3.12
import sys
from os import remove, popen
from os.path import exists, realpath, dirname
from os import getenv
from pytest import mark

@mark.skipif(sys.version_info >= (3, 14), 
                   reason="Skipping Keras tests for Python 3.14")
def test_shgo_shekel_onnx(monkeypatch, request):
    from shgo_onnx import main
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        onnx_model =  test_path + '/shekel_model.onnx'
        if exists(onnx_model):
            remove(onnx_model)
        assert exists(onnx_model) == False

        print("")
        m.setattr(sys, 'argv', ['shgo_onnx'])
        assert main(test_path) == '22db6896cbc7ce1cd90d0bc37d154b2f66eafe4da6f3447735e4fd1dd58f1a8c'
