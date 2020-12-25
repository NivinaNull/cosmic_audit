#-*- coding: utf-8 -*-
# @Time    : 11:17 
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : new_words
from smoothnlp.algorithm.phrase import extract_phrase
import pandas as pd
import re
from pathlib import Path
import sys

pd.set_option('display.max_columns',20) #给最大列设置为20列
pd.set_option('display.max_rows',10)#设置最大可见10行


def get_file():
    # cosmic_cols = ['type', 'batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'requirement_detail', 'advocator',
    #                                       'days_spent', 'coding_requirement']
    #
    # noncosmic_cols = ['type', 'batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'work_cat', 'work_name',
    #                                     'work_detail', 'days_spent']
    # maintenance_req_cols = ['type', 'projectNO', 'project_name', 'batch', 'requirementNO', 'requirement_name',
    #                   'requirement_detail', 'advocator', 'days_spent', 'phase_detail']
    raw_cosmic = pd.read_pickle('./data/cosmic_info.pkl').fillna('')  ##########.sample(n=10)
    raw_cosmic = raw_cosmic[['project_name', 'requirement_name', 'requirement_detail', 'advocator', 'coding_requirement']].astype(str)
    raw_noncosmic = pd.read_pickle('./data/noncosmic_info.pkl').fillna('')
    raw_noncosmic = raw_noncosmic[['project_name', 'requirement_name', 'work_cat', 'work_name', 'work_detail']].astype(str)
    maintenance_req = pd.read_pickle('./data/maintenance_req.pkl').fillna('')
    maintenance_req = maintenance_req[['project_name', 'requirement_name', 'requirement_detail', 'advocator', 'phase_detail']]
    return raw_cosmic, raw_noncosmic, maintenance_req


def get_corpus(df):
    corpus = []
    for index, r in df.iterrows():
        corpus.append(' '.join(r.values))
    return corpus


def discover_names(name_to_extract):
    name_set = set()
    name_list = []
    for n in name_to_extract:
        n = re.split('[ :：/\\\\]', n)[-1]
        p = re.compile(r'[\u4e00-\u9fa5]')
        n = re.findall(p, n)
        # 名字长度在2-4之间
        if 1 < len(n) < 5:
            name_set.add(''.join(n))
    his_name = []
    name_dict_location = Path('./data/name_dic.txt')
    if name_dict_location.is_file():
        with open(name_dict_location, 'r', encoding='utf-8') as f:
            try:
                for line in f.readlines():
                    if line:
                        print(line.split(' ')[0])
                        his_name.append(line.split(' ')[0])
            except Exception as e:
                print('原 name_dic.txt 无内容' + str(e))
    print('his_name_print ' * 8, his_name)
    with open('./data/name_dic.txt', 'a', encoding='utf-8') as f:
        for n in name_set:
            if n not in his_name:
                name_list.append(' '.join([n, '100', 'nr']))
        f.write('\n'.join(name_list))


def discover_words(corpus):
    new_phrases = extract_phrase(corpus, top_k=2000)
    new_phrases = [i + ' 100 n' for i in new_phrases ]

    with open('C:\\ChinaMobile\\new_words_detect.txt', 'w') as fout:
        fout.write('\n'.join(new_phrases))


if __name__ == '__main__':
    cosmic_df, noncosmic_df, maintenance_df = get_file()
    # name_corpus = cosmic_df['advocator'].to_list()
    # name_corpus.extend(maintenance_df['advocator'].to_list())
    # # discover_names(name_corpus)
    all_corpus = []
    for c in ['project_name', 'requirement_name', 'requirement_detail', 'coding_requirement']:
        all_corpus.append(cosmic_df[c].to_list())
    for c in ['project_name', 'requirement_name', 'work_cat', 'work_name', 'work_detail']:
        all_corpus.append(noncosmic_df[c].to_list())
    for c in ['project_name', 'requirement_name', 'requirement_detail', 'phase_detail']:
        all_corpus.append(maintenance_df[c].to_list())
    discover_words(all_corpus)
