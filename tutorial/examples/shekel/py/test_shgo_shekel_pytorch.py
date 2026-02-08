#!/usr/bin/python3.12
import sys
from os.path import realpath, dirname
from os import getenv
from pytest import mark

@mark.skipif(sys.version_info >= (3, 14), 
                   reason="Skipping ONNX tests for Python 3.14")
def test_shgo_shekel_onnx(monkeypatch, request):
    from shgo_onnx_pytorch import main
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        print("")
        m.setattr(sys, 'argv', ['shgo_onnx_pytorch'])
        assert main(test_path) == '79557cfaf4ead52dfa80a998e0e4161713eac5bbac383da1e9fea96fe4a1ed64'
