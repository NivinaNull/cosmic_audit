#-*- coding: utf-8 -*-
# @Time    : 2020/6/5 15:48
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : explore1.py

import pandas as pd
import datetime
import numpy as np

pd.set_option('display.max_columns',20) #给最大列设置为10列
pd.set_option('display.max_rows',100)#设置最大可见100行


# test_dict = {'a':[np.nan, np.nan, np.nan, np.nan], 'b':['g', 'h', 'f', 'i'], 'c':['gg', 'hh', 'ff', np.nan]}
# aa = pd.DataFrame(test_dict)
# # print(type(aa.b))
# # #  <class 'pandas.core.series.Series'>
# # print(type(aa.iloc(axis=1)[1]))
# # #  <class 'pandas.core.series.Series'>
# # print(type(aa['b']))
# # #  <class 'pandas.core.series.Series'>
# # print(type(aa[['b']]))
# # #  <class 'pandas.core.frame.DataFrame'>
#
# aa = aa.fillna('')
# print(aa)
# #   a  b   c
# # 0    g  gg
# # 1    h  hh
# # 2    f  ff
# # 3    i
#
# # aa['add1'] = aa.a + aa.b
# # aa['add2'] = aa.b + aa.c
# # print(aa)

# year_list = [str(i + datetime.datetime.now().year) for i in range(-20, 21)]
# year_list = year_list + [i for i in range(201)]

###############################################pandas Dataframe传入函数中会被改变#################################33
# import copy
# df_test = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#                    columns=['a', 'b', 'c'])
# print(df_test)
#
# def how_parameter(df):
#     df_inner = copy.deepcopy(df)
#     df_inner.drop(columns=['a'], inplace=True)
#     print(df_inner)
#
# how_parameter(df_test)
#
# print(df_test)
########################################################################################################################

def where_name():
    # raw_cosmic = pd.read_pickle('./data/cosmic_info_Q3.pkl').fillna('')
    # raw_noncosmic = pd.read_pickle('./data/noncosmic_info_Q3.pkl').fillna('')
    raw_cosmic = pd.read_csv('./data/cosmic_info_Q3.csv').fillna('')
    raw_noncosmic = pd.read_csv('./data/noncosmic_info_Q3.csv').fillna('')
    # raw_cosmic = raw_cosmic[['batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'requirement_detail', 'coding_requirement']]
    # raw_cosmic = raw_cosmic.astype(str)
    # for row in raw_cosmic.values:
    #     for c in range(7):
    #         if '孙全勇' in row[c]:
    #             print(row)
    # print(raw_noncosmic.shape)
    print(raw_cosmic)
    print(raw_noncosmic)

# # where_name()
# from pathlib import Path
# p1 = Path('C:\\ChinaMobile\\test_output\\正常文件夹\\╒²│ú╬─╝■╝╨')
# p2 = Path('C:\\ChinaMobile\\test_output\\正常文件夹\\正常文件夹')
# print(p1.replace(p2))
# p1.replace(p2)
# print(p1)
# cosmic_info = pd.read_pickle('./data/cosmic_info.pkl')
# print(cosmic_info[cosmic_info.projectNo == '101'])
# set1= set([1, 2, 3])
# set2= set([1, 2, 3, 4, 5])
# print(set1 - set2)
# df = pd.DataFrame([[0, 2, 3], [0, None, 1], [10, 20, 30]])
# print(df)
# print(df.dtypes)
# from pathlib import Path
# import zipfile
# with zipfile.ZipFile('./data/system_path.zip', 'r') as f:
#     for fn in f.namelist():
#         # f.extract(fn)
#         extracted_path = Path(f.extract(fn))    # f : zip_file
#         extracted_path.rename(fn.encode('cp437').decode('gbk'))
#         print(extracted_path)

# list_test = ['D:\\Audit\\专家评审材料\\2020-1-322-_集中化IOP平台开发服务项目.zip']
# print(list_test)


# print('D:\\Audit\\专家评审材料\\2020-1-322-_集中化IOP平台开发服务项目.zip'.encode('utf-8'))
# print('D:\\Audit\\专家评审材料\\2020-1-322-_集中化IOP平台开发服务项目.zip'.encode('utf-8').decode('utf-8'))

# 需求序号：001

# print(re.__version__)
# to_be_split = 'R-YOA2-CWGS-03财务公司组织机构、人员调整'

# print(re.search('^(.*需求.*|.*编号.*|.*序号.*|)' + '2', '共2个需求2'))
# print(format2(to_be_split))
#

# temp = "想做/ 兼_职/学生_/ 的 、加,我Q:  1 5.  8 0. ！！？？  8 6 。0.  2。 3     有,惊,喜,哦"
#
# string = re.sub("[\s+\.\!\/\:_,$%^*(+\"\']+|[+——！，。？、：~@#￥%……&*（）]+", "",temp)
# print(string)
# to_be = 'of需  求  序号202:[河北省公司-市场经营部-电子渠道中心-运营室]I申请开发2020年1月账单查询，并在和生活等渠道进行承载'
# m = re.search('^[(.*需求.*?)|(.*编号.*?)|(.*序号.*?)]202' , to_be)
# m = re.search('^.*序号.{0,?}|.*编号.{0,1}|.*需求.{0,1}202' , to_be)
# m = re.search('^(.*需求.*?|.*序号.*?|.*编号.*?)202' , to_be)
# print(m)


# import jieba
# print(jieba.lcut(to_be))
# print([ord(x) for x in '+-*/'])
# # what = re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])", 'Nivina99')
# # print(what)
# to_be_split = '2财务公司组织机构、人员调整'
# # to_be_split = 'R-YOA2-CWGS-03财务公司组织机构、人员调整'
# # t1 = re.search('^((\w+)-(\w+)\d|需求.*|编号.*|序号.*|2)', to_be_split)
# t1 = re.search('^(需求.*|编号.*|序号.*|)'+'2' , to_be_split)
# print(t1)
# # print(re.search('^(\W+)' + str(2),  to_be_split))
# s1 = pd.DataFrame(['    无  ', '无', '无关紧要'])
# print(s1)
# s1 = s1.applymap(lambda i: ' ' if i.strip() == '无' else i)
# print(s1)
# print(re.split(':|：|-', '需求序号：003-IT中心收文三合一（网关改造）'))
# temp = pd.DataFrame()
# temp['requirementNO'] = [25]
# print(temp)
# print(datetime.datetime.now().month - 1)
# import datetime
# current_year = datetime.datetime.now().year
# print(type(current_year))
# # df1 = pd.DataFrame()
# if not df1.empty:
#     print('its nonsense' * 10)
# print(df1.shape[1])

# if None:
#     print('1'*10)
# else:
#     print('2'*10)


# file_path = "./data/附件7：需求序号001：COSMIC软件评估功能点拆分表.xlsx"
# ws_name = '功能点拆分表'
# cosmic_metric = pd.DataFrame(columns=['batch', 'project', 'requirementNO', 'requirement_name', 'requirement_detail', 'advocator',
                                      # 'days_spent', 'coding_requirement'])


# file_path = 'E:\\organized_file\\专家评审材料\\附件9：工作量核算表（结算）-项目序号169.xls'
# all = pd.read_excel(file_path, sheet_name=None, header=None)
# s2_key = list(all.keys())[1]
# s2 = all[s2_key].drop(index=0).reset_index(drop=True)
# get_cosmic_info(s2)





# print(df1)
# print(type(df1))




# import os
# for folderName, subfolders, filenames in os.walk('E:\\test_folder'):
#     for filename in filenames:
#         # if filename.find('附件'):
#         if '附件' in filename:
#             print('it can unlock the compressed file')
#     print(folderName)
#     print(subfolders)
#     print(filenames)


############################################################################list中  bytes类型#################################
# import codecs
# stopwords = [line.decode('utf-8').strip() for line in codecs.open('./data/stopwords.txt', 'rb').readlines()]
# print('len(stopwords)'*5, len(stopwords))
# print(stopwords)
# def num_process(w):
#     if w in ['10086', '10085']:
#         return True
#     try:
#         int(w)
#         return False
#     except Exception as e:
#         print('in Exception ' * 5, w)
#         if w in stopwords:
#             print('in stopwords ' * 5, w)
#             return False
#         else:
#             return True
#
#
# words = ['的', '人天', '大数据', '中移IT', ' ']
# words = [i for i in words if i.strip() and num_process(i)]
# print(words)
###########################################################################################################################


def test_dict():
    # dict1 = {'a':5, 'b':6, 'c':1, 'd':3, 'e':2}
    # dict1 = sorted(dict1.items(), key=lambda x:x[1], reverse=True)

    dict2 = {('2020Q1', '20190103999', 7): 0.63366085, ('2020Q1', '20190103999', 58): 0.6445195}
    dict2 = sorted(dict2.items(), key=lambda x:x[1], reverse=True)
    print(dict2)

    # print(dict1)
    # for i in dict1.keys():
    #     print(i)
# test_dict()

# if np.nan:
#     print('something, anything')

# ser1 = pd.Series([np.nan, np.nan, np.nan, np.nan])
# print(ser1)
# print(ser1.astype(str))
# print(ser1.astype(str).str.strip())



# print(np.zeros(200, dtype=np.float32))

# df_from_np = pd.DataFrame(np.random.randn(3,4), columns=['one', 'two', 'three', 'four'])
# print(np.ones((3,5),dtype=float))
# df_from_np['three'] = np.ones((3,5),dtype=float)
# print(df_from_np['three'])
# df_from_np = pd.DataFrame(np.random.randn(3,4), columns=['one', 'two', 'three', 'four'])
# two_dimensions = np.random.randn(3,5)
# print(two_dimensions)
# raw_noncosmic = pd.read_pickle('./data/noncosmic_info.pkl').fillna('')
# print('raw_noncosmic.dtypes' * 10 + '\n', raw_noncosmic.dtypes)

# df_from_np['three'] = [np.zeros(20,dtype=float) for i in range(3)]
# print(df_from_np['three'].dtype)
#
# print(df_from_np)
# for i in df_from_np.iterrows():
#     print(i)
#     print(i[1]['three'])
#     i[1]['three'] = np.ones(20, dtype=np.float32)
#     print(i[1]['three'])
#
# print(df_from_np)


def cosine_distance(matrix1, matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    # print('1' * 10 + '\n', matrix1_matrix2)
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    # print('2' * 10 + '\n',matrix1_norm)
    matrix1_norm = matrix1_norm[:, np.newaxis]
    # print(matrix1_norm)
    # print(matrix2, '\n' + 'matrix2.shape ' * 8, matrix2.shape)
    # print('np.multiply(matrix2, matrix2) ' * 5 + '\n', np.multiply(matrix2, matrix2))
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    # print('3' * 10 + '\n',matrix2_norm)
    # print('np.sqrt(np.multiply(matrix2, matrix2).sum(axis=0)) ' * 5 + '\n', np.sqrt(np.multiply(matrix2, matrix2).sum(axis=0)))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    # print('4' * 10 + '\n',matrix2_norm)
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    # print('5' * 10 + '\n',cosine_distance)
    return cosine_distance

# matrix1=np.array([[1,1],[1,2]])
# matrix2=np.array([[2,1],[2,2],[2,3], [3,4]])
# cosine_dis = cosine_distance(matrix1,matrix2)
# pos = np.where(cosine_dis >= 0.98)
# print('position_ ' * 20 + '\n',  pos)
# sim_value = cosine_dis[pos]
# print(sim_value)
# sim_index = {}
# print (cosine_dis)

# raw_cosmic = pd.read_pickle('./data/cosmic_info.pkl').fillna('')
# result = {('2020Q1', '177', 77): {('2020Q1', '101', 10): 0.9633335924454794}}
# for k, item in result.items():
#     req_detail = raw_cosmic[(raw_cosmic.batch == k[0]) & (raw_cosmic.projectNo == k[1]) & (raw_cosmic.requirementNO == k[2])]
#     print(req_detail)


def test_generator():
    data_generator = (x * x for x in range(3))
    print(data_generator)
    for i in data_generator:
        print(i, end=' ')
    print()
    print('第二次迭代data_generator,什么都不会输出')
    print()
    print(data_generator)
    for i in data_generator:
        print(i, end=' ')

# test_generator()


from collections import OrderedDict
import json

def test_dict_values():
    a = OrderedDict([
        ('batch', '需求批次'),
        ('projectNo', '项目编号'),
        ('project_name', '项目名称'),
        ('requirementNO', '需求编号'),
        ('requirement_name', '需求名称'),
    ])
    print(a.values())
    # print(json.dumps(list(a.values())))
    print(json.dumps(['需求批次', '项目编号', '项目名称'], ensure_ascii=False))
    print(json.dumps({'batch': '需求批次', 'projectNo': '项目编号'}, ensure_ascii=False))
    print(list(a.values())[0])
    print('\t'.join(a.values()))
    print(type(a.values()))
    for i in a.values():
        print('#' * 50 , type(i))

# li = ['30(人天)', '什么玩意儿']
# #
# # print(list(map(lambda x: x.split('(人天)')[0] if type(x.split('(人天)')[0]) == float else -1, li)))
# #
# #
# # print('30(人天)'.split('(人天)')[0], type('30(人天)'.split('(人天)')[0]))


# df_animal = pd.DataFrame([('bird', 389.0, 1, 5),
#                    ('bird', 24.0, 6, 10),
#                    ('mammal', 80.5, 11, 15),
#                    ('mammal', np.nan, 16, 20)],
#                   index=['falcon', 'parrot', 'lion', 'monkey'],
#                   columns=('class', 'max_speed', 'start', 'end'))
# print(df_animal[(df_animal.start < 4) & (df_animal.end > 4) ])
# print(df_animal.reset_index(col_level=1))

# ani_list = ['falcon', 'parrot', 'lion', 'monkey']
# print(ani_list[:-1])
# print(ani_list[1:])

# import re
# if (re.findall("\d+", 22)):
#     print(re.findall("\d+", '88（人天）'))
#     print('it works ')
# else:
#     print('nothing good lasts forever')
#

import os,shutil
from pathlib import Path
# path = Path('C:\\ChinaMobile\\cosmic_files\\专家评审材料_2020Q3_auto\\p11\\2020-3-194-2019年中国移动内蒙古公司电子渠道系统业务需求技术服务框架（新北洋）\\2020-3-194-2019年中国移动内蒙古公司电子渠道系统业务需求技术服务框架（新北洋）\\专家评审材料三季度（新北洋项目序号194）\\专家评审材料三季度（新北洋项目序号194）\\系统序号194：互联网平台及自助终端系统2020年第三季度输出物\\关于三屏合一无法办理携转业务的优化需求：COSMIC软件评估功能点拆分表.xlsx')
# path = '\\\\?\\C:\\ChinaMobile\\cosmic_files\\专家评审材料_2020Q3_auto\\p11\\2020-3-194-2019年中国移动内蒙古公司电子渠道系统业务需求技术服务框架（新北洋）\\2020-3-194-2019年中国移动内蒙古公司电子渠道系统业务需求技术服务框架（新北洋）\\专家评审材料三季度（新北洋项目序号194）\\专家评审材料三季度（新北洋项目序号194）\\系统序号194：互联网平台及自助终端系统2020年第三季度输出物\\关于三屏合一无法办理携转业务的优化需求：COSMIC软件评估功能点拆分表.xlsx'
# test_path = '\\\\?\\C:\\ChinaMobile\\p1'
# test_path = '\\\\?\\C:\\ChinaMobile\\cosmic_files\\p1'
# os.makedirs(test_path)
# with open(test_path, 'wb'):
#     shutil.copyfileobj(source, test_path)
# with open(path, 'wb') as f:
#     f.write('something like this \n蝙蝠侠')
# path = win32api.GetShortPathName(path)

# with open(path, 'w') as f:
#     f.write('something like this \n poop')
#
# if os.name == 'nt':
#     print('WIN ' * 3)

