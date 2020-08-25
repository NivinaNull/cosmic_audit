#-*- coding: utf-8 -*-
# @Time    : 2020/8/4 10:28
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : similarity_analyse.py

import numpy as np
import pandas as pd
import datetime
import copy
import sys
import jieba
import jieba.posseg as pseg
import codecs
import re
from gensim import corpora, models
from gensim.similarities import SparseMatrixSimilarity


pd.set_option('display.max_columns',20) #给最大列设置为20列
pd.set_option('display.max_rows',10)#设置最大可见10行


def get_new_req(cosmic, noncosmic, flag):
    if flag == 'cosmic':
        return cosmic.sample(n=10)
    elif flag == 'noncosmic':
        return noncosmic.sample(n=10)


# result = get_sim(info[FLAG], new_req_corpus, flag)
# 已存需求的分词词料，新需求的分词词料，FLAG
def get_sim(previous_req, new_req, flag):
    # 利用cosmic和非cosmic信息构建的Dictionary
    dictionary = corpora.Dictionary.load_from_text('./data/total.dic')
    # print('new_req.shape：', new_req.shape, 'raw_info.shape:', raw_info.shape, 'previous_req.shape:', previous_req.shape)
    corpus = [dictionary.doc2bow(text) for text in previous_req[flag]]
    tfidf = models.TfidfModel(corpus)
    index = SparseMatrixSimilarity(tfidf[corpus], 3600)

    new = [dictionary.doc2bow(t) for t in new_req['joint_info']]
    sim_dict = {}
    sim = index[new]
    for i in range(new_req.shape[0]):
        key = (new_req['batch'].iloc[i], new_req['projectNo'].iloc[i], new_req['requirementNO'].iloc[i])
        value = {}
        # print(key)
        for j in range(len(sim[i])):
            if sim[i][j] >= 0.5:
                inner_key = (previous_req['batch'].iloc[j], previous_req['projectNo'].iloc[j], previous_req['requirementNO'].iloc[j])
                if inner_key == key:
                    continue
                else:
                    value[inner_key] = sim[i][j]
        if value:
            sim_dict[key] = value
    return sim_dict


def get_corpus_data(cosmic_info=pd.DataFrame(), noncosmic_info=pd.DataFrame(), result_type='total'):
    stopwords = [line.decode('utf-8').strip() for line in codecs.open('./data/stopwords.txt', 'rb').readlines()]
    def num_process(w):
        if w in ['10086', '10085']:
            return True
        try:
            int(w)
            return False
        except Exception as e:
            if w in stopwords:
                return False
            else:
                return True

    def segment(row):
        if row:
            year_list = [str(i + datetime.datetime.now().year) for i in range(-20, 21)]
            year_related = ['\d*' + y + '\d*' for y in year_list]
            year_related_str = '|'.join(year_related)
            row = re.sub(year_related_str + '|\d\.|\d、|\d,|\d，|\W+', ' ', row).replace('_', ' ')
            # row = re.sub('\d\.|\d、|\d|\n|\t', '', row)
            jieba.load_userdict('./data/my_dict.txt')
            words = jieba.lcut(row)
            # stop_num_list = [i for i in range(201)]
            words = [i for i in words if i.strip() and num_process(i)]
        else:
            words = []
        # return ' '.join(words)
        return words

    def to_str(attri_row):
        joint_info = ''
        for c in range(len(attri_row.work_detail)):
            if not (attri_row.work_detail[c].strip() == '无' or attri_row.work_detail[c].strip() == ''):
                joint_info = attri_row.work_detail[c]
                if not (attri_row.work_name[c].strip() == '无' or attri_row.work_name[c].strip() == ''):
                    joint_info = joint_info + ' ' + attri_row.work_name[c]
                if not (attri_row.work_cat[c].strip() == '无' or attri_row.work_cat[c].strip() == ''):
                    joint_info = joint_info + ' ' + attri_row.work_cat[c] + ' '
        return joint_info

    cosmic_inner = copy.deepcopy(cosmic_info)
    noncosmic_inner = copy.deepcopy(noncosmic_info)
    if not noncosmic_inner.empty:
        noncosmic_inner['joint_info'] = noncosmic_inner[[ 'work_cat', 'work_name', 'work_detail']].apply(to_str, axis=1 )
        noncosmic_inner['joint_info'] = noncosmic_inner[ 'joint_info'] + noncosmic_inner['requirement_name']
        if result_type == 'noncosmic':
            noncosmic_inner['joint_info'] = noncosmic_inner['joint_info'].map(segment)
            return True, noncosmic_inner
        elif result_type == 'total':
            noncosmic_inner.drop(
                columns=['work_cat', 'work_name', 'work_detail', 'requirement_name', 'project_name', 'days_spent'], inplace=True)
            noncosmic_inner.rename(columns={'joint_info':'noncosmic'}, inplace=True)
        else:
            return False, 'ERROR:您传入了不需要的noncosmic信息，请检查'

    if not cosmic_inner.empty:
        cosmic_inner['joint_info'] = ''
        for i in ['project_name', 'requirement_name', 'requirement_detail', 'coding_requirement']:
            cosmic_inner['joint_info'] = cosmic_inner['joint_info'] + ' ' + cosmic_inner[i]
        if result_type == 'cosmic':
            cosmic_inner['joint_info'] = cosmic_inner['joint_info'].map(segment)
            return True, cosmic_inner
        elif result_type == 'total':
            cosmic_inner.rename(columns={'joint_info':'cosmic'}, inplace=True)
            cosmic_inner = cosmic_inner[['batch', 'projectNo', 'requirementNO', 'cosmic']]
            total = pd.merge(cosmic_inner, noncosmic_inner, how='outer', on=['batch', 'projectNo', 'requirementNO']).fillna('')
            total['combo'] = total['cosmic'] + ' ' + total['noncosmic']
            total[['cosmic', 'noncosmic', 'combo']] = total[['cosmic', 'noncosmic', 'combo']].applymap(segment)
            return True, total
        else:
            return False, 'ERROR:您传入了不需要的cosmic信息，请检查'
    return False, 'ERROR：未发现有效数据'


def show(result_dict, raw_info, new, flag):
    # print(result_dict)
    if result_dict:
        for k in result_dict.keys():
            req_detail = new[(new.batch == k[0]) & (new.projectNo == k[1]) & (new.requirementNO == k[2])]
            print('发现' + flag + '需求批次' + k[0] + '项目编号' + k[1] + '需求编号' + str(k[2]) + '\n', '需求详细信息：', req_detail.to_json(force_ascii=False, orient='split', index=False), '相似需求如下：')
            # print('result_dict[k]111' * 5, result_dict[k])
            result_dict[k] = sorted(result_dict[k].items(), key=lambda x: x[1], reverse=True)
            # print('result_dict[k]222' * 5, result_dict[k])
            for ki in result_dict[k]:
                sim_detail = raw_info[(raw_info.batch == ki[0][0]) & (raw_info.projectNo == ki[0][1]) & (raw_info.requirementNO == ki[0][2])]
                print('相似需求批次' + ki[0][0] + '项目编号' + ki[0][1] + '需求编号' + str(ki[0][2]) + '\n', sim_detail.to_json(force_ascii=False, orient='split', index=False))
                print('相似度为：', ki[1])
    else:
        print('未发现相似' + flag + '需求')


if __name__ == '__main__':
    raw_cosmic = pd.read_pickle('./data/cosmic_info.pkl').fillna('')
    raw_noncosmic = pd.read_pickle('./data/noncosmic_info.pkl').fillna('')
    SUCCESS, info = get_corpus_data(raw_cosmic, raw_noncosmic, 'total')


    if not SUCCESS:
        print(info)
        sys.exit()
    # info.to_csv('./data/info.csv', encoding="utf_8_sig", index=False)
    dictionary = corpora.Dictionary(info.combo.to_list())
    # print('dictionary.num_pos'*5, dictionary.num_pos)
    dictionary.filter_extremes(no_below=1, no_above=0.9, keep_n=3600)

    ####################################################check the dictionary###########################

    # print('len(dictionary.cfs)' * 5, len(dictionary.cfs))  ###### 8426

    # #########   所有词的个数（不去重）310509
    # sum of the number of unique words per documen over the entire corpus
    # print('dictionary.num_nnz' * 5, dictionary.num_nnz)

    # cfs = dictionary.cfs
    # token_fs = {}
    # for t, id in dictionary.token2id.items():
    #     # print(t, id)
    #     token_fs[t] = cfs[id]
    # # d1 = sorted(dict.items(), key=lambda x: x[0], reverse=False)
    # token_fs = sorted(token_fs.items(), key=lambda x:x[1], reverse=True)
    # # print('token_fs' * 10 + '\n', token_fs)
    ####################################################################################################

    dictionary.save_as_text('./data/total.dic')

    cosmic_sim_analyse = []
    noncosmic_sim_analyse = []
    for FLAG in ['cosmic', 'noncosmic']:
        fraud_new_req = get_new_req(raw_cosmic, raw_noncosmic, flag=FLAG)
        if fraud_new_req.empty:
            print('ERROR：新读入的' + FLAG + '需求为空')
            continue
        else:

            if FLAG == 'cosmic':
                raw_info = raw_cosmic
                SUCCESS, new_req_corpus = get_corpus_data(cosmic_info=fraud_new_req, result_type=FLAG)
            else:
                raw_info = raw_noncosmic
                SUCCESS, new_req_corpus = get_corpus_data(noncosmic_info=fraud_new_req, result_type=FLAG)
            if SUCCESS:
                solo = info[FLAG].apply(lambda x:np.NAN if len(x) == 0 else x).dropna()
                solo = info.iloc[solo.index][['batch', 'projectNo', 'requirementNO',FLAG]]
                sim_result = get_sim(solo, new_req_corpus, flag=FLAG)
                show(sim_result, raw_info, fraud_new_req, FLAG)
            else:
                continue