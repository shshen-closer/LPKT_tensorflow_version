import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import os
import sys
import time, datetime
from sklearn.model_selection import train_test_split

#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)

#len: 755442
##length: 97.40831063139196
#its: 72830887
#its: 41829
#ats: 73586329
#ats: 9257
question2answer = {}
question2kc = {}
question2bundle = {}

problem2id = {}
at2id = {}
it2id = {}
skill2id = {}
bundle2id = {}
kcs = []
bundles = []
questions_data = pd.read_csv('data/contents/questions.csv', encoding = "ISO-8859-1", low_memory=False)
questions_data = np.array(questions_data)
count = 1
for line in questions_data:
    
    item = line[5].strip().split(';')
    item = [int(x) for x in item]
    if -1 in item:
        continue
    item = item[0]
    question2answer[line[0]] = line[3]
    problem2id[line[0]] = count
    count += 1
    bundles.append(line[1])
    kcs.append(item)
kc2id = {}
kcs = set(kcs)
print(len(kcs))
bundles = set(bundles)
count = 0
for i in bundles:
    bundle2id[i] = count 
    count += 1

count = 0
for i in kcs:
    kc2id[i] = count 
    count += 1

count = 1
for line in questions_data:
    item = line[5].strip().split(';')
    item = [int(x) for x in item]

    
    if -1 in item:
        continue
    item = item[0]
    
    item = kc2id[item]
    
    question2kc[line[0]] = item
    question2bundle[line[0]] = bundle2id[line[1]]
    count += 1

with open('data/question2bundle', 'w', encoding = 'utf-8') as fo:
    fo.write(str(question2bundle))
print(len(bundles))
print(len(kcs))
print(len(question2answer))

it_id = []
at_id = []
file_path = 'data/KT1/'
list_file = os.listdir(file_path)
length = []
for iii in tqdm(list_file):    
    one_data =  pd.read_csv(file_path + iii, encoding = "ISO-8859-1", low_memory=False)
    temp1 = np.array(one_data)
    temp1 = [x for x in temp1 if x[2] in question2kc.keys()]


    temp = np.array(temp1)

    if len(temp) < 2:
        continue
    length.append(len(temp))
    at_id.append(temp[0][-1] / 1000)
    for iii in range(1, len(temp)):
        at_id.append(temp[iii][-1] / 1000)
        a = int((temp[iii][0] - temp[iii-1][0]) / 60000)
        if a > 43200:
            a = 43200
        it_id.append(a)
print('len:',  len(length))
print('length:',  np.mean(length))
print('its:',  len(it_id))
it = set(it_id) 
print('its:',  len(it))
it2id = {}

count = 1
for i in it:
    it2id[i] = count 
    count += 1

print('ats:',  len(at_id))
at = set(at_id) 
print('ats:',  len(at))
at2id = {}

count = 1
for i in at:
    at2id[i] = count 
    count += 1
with open('data/it2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(it2id))
with open('data/at2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(at2id))




with open('data/problem2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(problem2id))
with open('data/skill2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(skill2id))

with open('data/question2kc', 'w', encoding = 'utf-8') as fo:
    fo.write(str(question2kc))
with open('data/question2answer', 'w', encoding = 'utf-8') as fo:
    fo.write(str(question2answer))
print('complete')
