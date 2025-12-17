#!/usr/bin/python3.12
import sys
from bnh import main
from os import remove, popen
from os.path import exists
from sys import version

def test_constraint_dora(monkeypatch):
    with monkeypatch.context() as m:
        out = 'NSGA2_pareto.txt'
        if exists(out):
            remove(out)
        assert exists(out) == False
        m.setattr(sys, 'argv', ['bnh'])
        assert main() == 0
        if version.split()[0] == '3.14.2':
            assert int(popen(f"sum {out}").read().split()[0]) == 19751
        else:
            assert int(popen(f"sum {out}").read().split()[0]) == 43431
