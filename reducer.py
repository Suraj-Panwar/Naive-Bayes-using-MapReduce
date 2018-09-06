#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None
current_label = None
dict_label = {}

# Take Serial data coming from mapper using sys
for line in sys.stdin:
    line = line.strip()

    label, word, count = line.split('\t')

    try:
        count = int(count)
    except ValueError:
        continue
    # Ignore non numeric cases
    # Generate Dictionary
    if label == current_label:
	if word in dict_label:
	   dict_label[word] += 1
	else:
	   dict_label[word] = 1

    # Output word data corresponding to label
    elif label != current_label and current_label != None:
	current_label = label
	for keys in dict_label.keys():
	   print('%s\t%s\t%s' %(current_label,keys,dict_label[keys]))
	dict_label = {}
    
    elif current_label == None:
	current_label = label

for keys in dict_label.keys():
	print('%s\t%s\t%s' %(current_label,keys,dict_label[keys]))


