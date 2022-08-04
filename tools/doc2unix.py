#!/usr/bin/env python
"""
Followed the instructions from  vkaushik189
https://github.com/udacity/ud120-projects/issues/181
to fix a bug

convert dos linefeeds (crlf) to unix (lf)
usage: python dos2unix.py
"""

import sys

original = 'word_data.pkl'
destination = "word_data_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))