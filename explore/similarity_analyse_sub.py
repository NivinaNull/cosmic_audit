#-*- coding: utf-8 -*-
# @Time    : 2020/8/4 10:28
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : similarity_analyse.py

import pandas as pd
import datetime
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity


from gensim import corpora, models
from gensim.similarities import SparseMatrixSimilarity

import jieba
import jieba.posseg as pseg
import codecs
import re

pd.set_option('display.max_columns',20) #给最大列设置为20列
pd.set_option('display.max_rows',10)#设置最大可见10行


def get_sim(data_info):
    # tfidf_model = TfidfVectorizer(min_df=2)
    # result = tfidf_model.fit_transform(data_info['combo'])
    # # for l in result.toarray():
    # #     print(len(l))
    # #     print([i for i in l if i != 0])
    # print(data_info.shape)
    # print(result.shape)
    # # print(result.toarray())
    # data_info['tfidf_weight'] = result
    # print(tfidf_model.get_feature_names())
    # print(len(tfidf_model.get_feature_names()))

    # vectorizer = CountVectorizer(max_features=4000)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(max_features=3600)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    count_vectorizer = vectorizer.fit_transform(data_info['combo'])
    count_feature_array = count_vectorizer.toarray()
    print(count_feature_array.shape)
    # print(vectorizer.vocabulary_)
    vocab_dict = vectorizer.vocabulary_
    vocab_fre_dict = {}
    for w, i in vocab_dict.items():
        num = 0
        # print(count_feature_array.shape[0])
        for j in range(count_feature_array.shape[0]):
            # print(count_feature_array[j][i])
            num += count_feature_array[j][i]
        vocab_fre_dict[w] = num
    vocab_fre_dict = sorted(vocab_fre_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(vocab_fre_dict)

    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(count_vectorizer)  # 内层fit_transform是将文本转为词频矩阵


    # cosine_similarity(a)

    # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # word = vectorizer.vocabulary_
    # print(len(word))
    # print(word)
    # weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     print("-------这里输出第", i, u"类文本的词语tf-idf权重------")
    #     for j in range(len(word)):
    #         print(word[j], weight[i][j])


def get_data():

    def segment(row):
        stopwords = [line.strip() for line in codecs.open('./data/stopwords.txt', 'rb').readlines()]

        def word_process(w):
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

        if row:
            year_list = [str(i + datetime.datetime.now().year) for i in range(-20, 21)]
            year_related = ['\d*' + y + '\d*' for y in year_list]
            year_related_str = '|'.join(year_related)
            row = re.sub(year_related_str + '|\d\.|\d、|\d,|\d，|\W+', ' ', row).replace('_', ' ')
            # row = re.sub('\d\.|\d、|\d|\n|\t', '', row)
            words = jieba.lcut(row)
            # stop_num_list = [i for i in range(201)]
            words = [i for i in words if i.strip() and word_process(i)]
            # print(words)
        else:
            words = []
        return ' '.join(words)

    def to_str(attri_row):
        joint_info = ''
        for c in range(len(attri_row.work_detail)):
            # print(attri_row.work_cat[c])
            if not (attri_row.work_detail[c].strip() == '无' or attri_row.work_detail[c].strip() == ''):
                joint_info = attri_row.work_detail[c]
                if not (attri_row.work_name[c].strip() == '无' or attri_row.work_name[c].strip() == ''):
                    joint_info = joint_info + ' ' + attri_row.work_name[c]
                if not (attri_row.work_cat[c].strip() == '无' or attri_row.work_cat[c].strip() == ''):
                    joint_info = joint_info + ' ' + attri_row.work_cat[c] + ' '
        return joint_info

    noncosmic_info = pd.read_pickle('./data/noncosmic_info.pkl').fillna('')
    noncosmic_info['joint_info'] = noncosmic_info[[ 'work_cat', 'work_name', 'work_detail']].apply(to_str, axis=1 )
    noncosmic_info['joint_info'] = noncosmic_info[ 'joint_info'] + noncosmic_info['requirement_name']
    noncosmic_info.drop(columns=['work_cat', 'work_name', 'work_detail', 'requirement_name', 'project_name', 'days_spent'], inplace=True)

    cosmic_info = pd.read_pickle('./data/cosmic_info.pkl').fillna('')
    cosmic_info['cosmic'] = ''
    for i in ['project_name', 'requirement_name', 'requirement_detail', 'advocator', 'coding_requirement']:
        cosmic_info['cosmic'] = cosmic_info['cosmic'] + ' ' + cosmic_info[i]
    cosmic_info = cosmic_info[['batch', 'projectNo', 'requirementNO', 'cosmic']]
    total = pd.merge(cosmic_info, noncosmic_info, how='outer', on=['batch', 'projectNo', 'requirementNO']).fillna('')
    total['combo'] = total['cosmic'] + ' ' + total['joint_info']
    total[['cosmic', 'joint_info', 'combo']] = total[['cosmic', 'joint_info', 'combo']].applymap(segment)
    return total


if __name__ == '__main__':
    info = get_data()
    info.to_csv('./data/info.csv', encoding="utf_8_sig", index=False)
    # get_sim(info)