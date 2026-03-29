#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run smlprover (SMLP)
#

# coding: utf-8

import sys, os

def main(args = None):
    smlpInst = SmlpFlows(args)
    smlpInst.smlp_flow()

if __name__ == '__main__':
    from smlp_py.smlp_flows import SmlpFlows
    main()
else:
    from .smlp_py.smlp_flows import SmlpFlows

