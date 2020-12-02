#-*- coding: utf-8 -*-
# @Time    : 11:17 
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : new_words
from smoothnlp.algorithm.phrase import extract_phrase
import pandas as pd
import sys

pd.set_option('display.max_columns',20) #给最大列设置为20列
pd.set_option('display.max_rows',10)#设置最大可见10行

raw_cosmic = pd.read_pickle('./data/cosmic_info_Q3.pkl').fillna('')  ##########.sample(n=10)
raw_cosmic = raw_cosmic[['project_name', 'requirement_name', 'requirement_detail', 'advocator', 'coding_requirement']].astype(str)
raw_noncosmic = pd.read_pickle('./data/noncosmic_info_Q3.pkl').fillna('')
raw_noncosmic = raw_noncosmic[['project_name', 'requirement_name', 'work_cat', 'work_name', 'work_detail']].astype(str)
# cosmic_cols = ['batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'requirement_detail', 'advocator',
#                                       'days_spent', 'coding_requirement']
#
# noncosmic_cols = ['batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'work_cat', 'work_name',
#                                     'work_detail', 'days_spent']


corpus = []
for index, r in raw_cosmic.iterrows():
    # print(r)
    # print(r.values)
    # print(len(r.values))
    corpus.append(' '.join(r.values))

for index, r in raw_noncosmic.iterrows():
    corpus.append(' '.join(r.values))


new_phrases = extract_phrase(corpus, top_k=1000)

with open('C:\\ChinaMobile\\new_words_detect.txt', 'w') as fout:
    fout.write('\t'.join(new_phrases))