# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:53:27 2017

@author: rmorriss
"""

users = dict()
print(type(users))

users['id'] = 1
users['name'] = 'Bman'

print(users.items())
print(users)

for varia, stuff in users.items():
    if varia == 'id':
        print(stuff)
        
newlist = []

users = [
{"id": 0, "name": "Hero"},
{"id": 1, "name": "Dunn"},
{"id": 2, "name": "Sue"},
]

for dicts in users:
    print(dicts.keys())
    print(dicts.values())
    print(dicts.items())
    
from collections import Counter
line = "I am a funny guy aren't I?"
list2 = []
words = line.split()
for word in words:
    word = word.lower()    
    word = word.strip(',!.?')
    print(word)
    list2.append(word)
word_counts = Counter(list2)
print(word_counts)

import numpy as np
a = np.arange(1,200,2).reshape(25,4)
print(a)

