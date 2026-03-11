#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run smlprover (SMLP)
#

# coding: utf-8

import sys, os

def main(argv):
    smlpInst = SmlpFlows(argv)
    smlpInst.smlp_flow()

if __name__ == '__main__':
    # TODO: this branch should go away: replace by invoking the 'smlp' script
    # which loads this file as a module
    from smlp_py.smlp_flows import SmlpFlows
    main(sys.argv)
else:
    from .smlp_py.smlp_flows import SmlpFlows
