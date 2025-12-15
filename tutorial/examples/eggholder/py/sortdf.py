#!/usr/bin/python3.14
import sys
from os.path import realpath, basename
from rich import print as rprint
import pandas as pd
import argparse
script_name = basename(realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', default="/dev/stdin")
parser.add_argument('--out',  '-o', default=None)
parser.add_argument('--column', '-c', type=int, default=1)
parser.add_argument('--header', '-hdr', default=None, action='store_true')
parser.add_argument('--reverse', '-r', default=False, action='store_true')
parser.add_argument('--separator', '-s', default='\\s+')
args=parser.parse_args()
try:
    df = pd.read_csv(args.file,sep=args.separator) \
         if args.header else pd.read_csv(args.file,sep=args.separator,header=None) 
    r = df.sort_values(by = df.columns[args.column-1], ascending = not args.reverse).values.tolist() 
    if args.out != None:
        original_stdout = sys.stdout
        outf=open(args.out,'w')
        sys.stdout=outf
    [print(*row, sep=' ') for row in r]
    if args.out != None:
        sys.stdout=original_stdout
        outf.close()
        
except Exception as err:
    rprint("\n[magenta] " + script_name + ":", "[red] ERROR: [/red]", "[red] " + str(err), "\n")
    exit(1)
