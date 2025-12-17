#!/usr/bin/python3.12
import sys
from constraint_dora import main
from os import remove, popen
from os.path import exists

def test_constraint_dora(monkeypatch):
    with monkeypatch.context() as m:
        out = 'dataset.txt'
        if exists(out):
            remove(out)
        assert exists(out) == False
        m.setattr(sys, 'argv', ['constraint_dora'])
        assert main() == 'a757d8ff6f40eacdbc7de3dd6c8afae2041f9bc2c7cd706ee0f0660f59d33803'
        assert int(popen(f"sum {out}").read().split()[0]) == 23847
