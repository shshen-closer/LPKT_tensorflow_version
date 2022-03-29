import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import sys
import time, datetime
from sklearn.model_selection import train_test_split, KFold

#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)
kfold = KFold(n_splits=5, shuffle=False)

length = int(sys.argv[1])

    
all_data =  pd.read_csv('data/2012-2013-data-with-predictions-4-final.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
all_data['timestamp'] =  all_data['end_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
all_data['answer_time'] =  all_data['end_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S'))) - all_data['start_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
order = ['user_id','problem_id','correct','skill_id', 'timestamp', 'answer_time']
all_data = all_data[order]
all_data['skill_id'].fillna('nan',inplace=True)
all_data = all_data[all_data['skill_id'] != 'nan'].reset_index(drop=True)


with open('data/at2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        at2id = eval(line)
with open('data/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/it2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        it2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)
user_id = np.array(all_data['user_id'])
user = list(set(user_id))
train_all_id, test_id = train_test_split(user,test_size=0.2,random_state=5)
train_all_id = np.array(train_all_id)
count = 0
for (train_index, valid_index) in kfold.split(train_all_id):
    
    train_id = train_all_id[train_index]
    valid_id = train_all_id[valid_index]
    np.random.shuffle(train_id)
    q_a_train = []

    for item in tqdm(train_id):

        idx = all_data[(all_data.user_id==item)].index.tolist() 
        temp1 = all_data.iloc[idx]
        temp1 = temp1.sort_values(by=['timestamp']) 
        temp = np.array(temp1)
        if len(temp) < 2:
            continue

        while len(temp) >= 2:
            quiz = temp[0:length]

            train_at = [at2id[int(quiz[0][5])]]
            train_it = [0]
            train_q = [problem2id[quiz[0][1]]]
            train_a = [int(quiz[0][2])]
            train_skill = [skill2id[quiz[0][3]]]

            for one in range(1,len(quiz)):
                
                train_at.append(at2id[int(quiz[one][5])])
                a = int((quiz[one][4] - quiz[one-1][4])/60)
                if a > 43200:
                    a = 43200
                train_it.append(it2id[a])
                train_q.append(problem2id[quiz[one][1]])
                train_a.append(int(quiz[one][2]))
                train_skill.append(skill2id[quiz[one][3]])

            q_a_train.append([train_at, train_it, train_q, train_a, train_skill, len(train_q)])
            temp = temp[length:]

    q_a_valid = []
    for item in tqdm(valid_id):

        idx = all_data[(all_data.user_id==item)].index.tolist() 
        temp1 = all_data.iloc[idx]
        temp1 = temp1.sort_values(by=['timestamp']) 
        temp = np.array(temp1)
        if len(temp) < 2:
            continue
        while len(temp) >= 2:
            quiz = temp[0:length]

            test_at = [at2id[int(quiz[0][5])]]
            test_it = [0]
            test_q = [problem2id[quiz[0][1]]]
            test_a = [int(quiz[0][2])]
            test_skill = [skill2id[quiz[0][3]]]

            for one in range(1,len(quiz)):

                test_at.append(at2id[int(quiz[one][5])])
                a = int((quiz[one][4] - quiz[one-1][4])/60)
                if a > 43200:
                    a = 43200
                test_it.append(it2id[a])
                test_q.append(problem2id[quiz[one][1]])
                test_a.append(int(quiz[one][2]))
                test_skill.append(skill2id[quiz[one][3]])

            q_a_valid.append([test_at, test_it, test_q, test_a, test_skill, len(test_q)])
            temp = temp[length:]
    np.random.seed(10)
    np.random.shuffle(q_a_train)
    np.random.seed(10)
    np.random.shuffle(q_a_valid)
    np.save("data/train" + str(count) + ".npy",np.array(q_a_train))

    np.save("data/test" + str(count) + ".npy",np.array(q_a_valid))

    count += 1
q_a_test = []
for item in tqdm(test_id):

    idx = all_data[(all_data.user_id==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
    while len(temp) >= 2:
        quiz = temp[0:length]

        test_at = [at2id[int(quiz[0][5])]]
        test_it = [0]
        test_q = [problem2id[quiz[0][1]]]
        test_a = [int(quiz[0][2])]
        test_skill = [skill2id[quiz[0][3]]]

        for one in range(1,len(quiz)):

            test_at.append(at2id[int(quiz[one][5])])
            a = int((quiz[one][4] - quiz[one-1][4])/60)
            if a > 43200:
                a = 43200
            test_it.append(it2id[a])
            test_q.append(problem2id[quiz[one][1]])
            test_a.append(int(quiz[one][2]))
            test_skill.append(skill2id[quiz[one][3]])

        q_a_test.append([test_at, test_it, test_q, test_a, test_skill, len(test_q)])
        temp = temp[length:]
np.random.seed(10)
np.random.shuffle(q_a_test)
np.save("data/test.npy",np.array(q_a_test))

print('complete')
            



 