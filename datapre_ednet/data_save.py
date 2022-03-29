import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import os
import sys
import time, datetime
from sklearn.model_selection import train_test_split, KFold
seq_len= int(sys.argv[1])
#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)
kfold = KFold(n_splits=5, shuffle=False)

with open('data/at2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        at2id = eval(line)
with open('data/question2kc', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        question2kc = eval(line)
with open('data/question2answer', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        question2answer = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/it2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        it2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)

length = []
file_path = 'data/KT1/'
list_file = os.listdir(file_path)
train_all_id, test_id = train_test_split(list_file,test_size=0.2,random_state=5)
train_all_id = np.array(train_all_id)
count = 0
for (train_index, valid_index) in kfold.split(train_all_id):
    train_id = train_all_id[train_index]
    valid_id = train_all_id[valid_index]
    np.random.shuffle(train_id)
    q_a_train = []
    for iii in tqdm(train_id):    
        one_data =  pd.read_csv(file_path + iii, encoding = "ISO-8859-1", low_memory=False)
        temp = np.array(one_data)
        temp = [x for x in temp if x[2] in question2kc.keys()]
        temp = np.array(temp)
        length.append(len(temp))

    #timestamp,solving_id,question_id,user_answer,elapsed_time
        if len(temp) < 2:
            continue
        while len(temp) >= 2:
            quiz = temp[0:seq_len]

            train_at = [at2id[quiz[0][4]/1000]]
            train_it = [0]
            train_q = [problem2id[quiz[0][2]]]
            train_a = [int(quiz[0][3] == question2answer[quiz[0][2]])]
            train_skill = [question2kc[quiz[0][2]]]

                
            for one in range(1,len(quiz)):
                train_at.append(at2id[quiz[one][4]/1000])
                if quiz[one][0] - quiz[one-1][0] < 0:
                    print('error')
                a = int((quiz[one][0] - quiz[one-1][0])/60000)
                if a > 43200:
                    a = 43200
                train_it.append(it2id[a])
                train_q.append(problem2id[quiz[one][2]])
                train_a.append(int(quiz[one][3] == question2answer[quiz[one][2]]))
                train_skill.append(question2kc[quiz[one][2]])

            q_a_train.append([train_at, train_it, train_q, train_a, train_skill, len(quiz)])
            temp = temp[seq_len:]

    q_a_valid = []
    for iii in tqdm(valid_id):    
        one_data =  pd.read_csv(file_path + iii, encoding = "ISO-8859-1", low_memory=False)
        temp = np.array(one_data)
        temp = [x for x in temp if x[2] in question2kc.keys()]
        temp = np.array(temp)
        length.append(len(temp))
    #timestamp,solving_id,question_id,user_answer,elapsed_time
        if len(temp) < 2:
            continue
        while len(temp) >= 2:
            quiz = temp[0:seq_len]

            test_at = [at2id[quiz[0][4]/1000]]
            test_it = [0]
            test_q = [problem2id[quiz[0][2]]]
            test_a = [int(quiz[0][3] == question2answer[quiz[0][2]])]
            test_skill = [question2kc[quiz[0][2]]]

                
            for one in range(1,len(quiz)):
                test_at.append(at2id[quiz[one][4]/1000])
                if quiz[one][0] - quiz[one-1][0] < 0:
                    print('error')
                a = int((quiz[one][0] - quiz[one-1][0])/60000)
                if a > 43200:
                    a = 43200
                test_it.append(it2id[a])
                test_q.append(problem2id[quiz[one][2]])
                test_a.append(int(quiz[one][3] == question2answer[quiz[one][2]]))
                test_skill.append(question2kc[quiz[one][2]])

            q_a_valid.append([test_at, test_it, test_q, test_a, test_skill, len(quiz)])
            temp = temp[seq_len:]
    np.save("data/train" + str(count) + ".npy",np.array(q_a_train))

    np.save("data/test" + str(count) + ".npy",np.array(q_a_valid))

    count += 1
    break

np.save("data/length.npy",np.array(length))


q_a_test = []
for iii in tqdm(test_id):    
    one_data =  pd.read_csv(file_path + iii, encoding = "ISO-8859-1", low_memory=False)
    temp = np.array(one_data)
    temp = [x for x in temp if x[2] in question2kc.keys()]
    temp = np.array(temp)
    length.append(len(temp))
#timestamp,solving_id,question_id,user_answer,elapsed_time
    if len(temp) < 2:
        continue
    while len(temp) >= 2:
        quiz = temp[0:seq_len]

        test_at = [at2id[quiz[0][4]/1000]]
        test_it = [0]
        test_q = [problem2id[quiz[0][2]]]
        test_a = [int(quiz[0][3] == question2answer[quiz[0][2]])]
        test_skill = [question2kc[quiz[0][2]]]

            
        for one in range(1,len(quiz)):
            test_at.append(at2id[quiz[one][4]/1000])
            if quiz[one][0] - quiz[one-1][0] < 0:
                print('error')
            a = int((quiz[one][0] - quiz[one-1][0])/60000)
            if a > 43200:
                a = 43200
            test_it.append(it2id[a])
            test_q.append(problem2id[quiz[one][2]])
            test_a.append(int(quiz[one][3] == question2answer[quiz[one][2]]))
            test_skill.append(question2kc[quiz[one][2]])

        q_a_test.append([test_at, test_it, test_q, test_a, test_skill, len(quiz)])
        temp = temp[seq_len:]

np.save("data/test.npy",np.array(q_a_test))

print('complete')
