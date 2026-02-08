#!/usr/bin/env python3
"""
wget replacement using Python's urllib
Usage: python-wget.py <url> [output_filename]
"""
import sys
import urllib.request
import os

if len(sys.argv) < 2:
    print("Usage: python-wget.py <url> [output_filename]")
    sys.exit(1)

url = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(url)

print(f"Downloading {url} to {output}...")
urllib.request.urlretrieve(url, output)
print(f"Download complete: {output}")
