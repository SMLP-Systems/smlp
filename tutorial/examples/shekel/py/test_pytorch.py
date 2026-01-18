#!/usr/bin/python3.12
import sys
from os import remove, popen
from os import getenv
from os.path import realpath,dirname,exists
from pytest import mark

@mark.skipif(sys.version_info >= (3, 14), 
        reason="Skipping PyTorch tests for Python 3.14")
def test_pytorch(monkeypatch, request):
    from pytorch_ex import main
    root_dir = str(request.config.rootpath) + '/'
    with monkeypatch.context() as m:
        test_path = dirname(realpath(root_dir + getenv('PYTEST_CURRENT_TEST').split(':')[0]))
        out  = test_path + '/scaler_X_pytorch.pkl'
        out1 = test_path + '/scaler_y_pytorch.pkl'
        out2 = test_path + '/shekel_model.onnx'
        for o in [out, out1]:
            if exists(out):
                remove(out)
        assert exists(out) == False
        print("")
        m.setattr(sys, 'argv', ['pytorch_ex'])
        assert main(512, test_path) == '0270d3c18ee4cf5a9b64b3ad80f9f3cc2886c97d5d7c2b842303c54bea569d6e'
        assert int(popen(f"sum {out}").read().split()[0])  == 27475
        assert int(popen(f"sum {out1}").read().split()[0]) == 24955
        assert int(popen(f"sum {out2}").read().split()[0]) == 36979
