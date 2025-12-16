#!/usr/bin/python3.12
import sys
from optimization_ex import main
from os import remove, popen
from os.path import exists

def test_optimization_ex(monkeypatch):
    with monkeypatch.context() as m:
        out = 'dataset.txt'
        if exists(out):
            remove(out)
        assert exists(out) == False
        m.setattr(sys, 'argv', ['optimization_ex'])
        assert main() == '54e5105d59a57fd2898e581ca6f1e3502d4cda22b371fa17a88420d6da862602'
        assert int(popen(f"sum {out}").read().split()[0]) == 35930
