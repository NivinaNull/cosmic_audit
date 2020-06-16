#-*- coding: utf-8 -*-
# @Time    : 2020/6/15 10:26
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : file_regularization.py

import shutil, os

def file_search(folderName, filenames):
    for filename in filenames:
        # print(filename)
        if '工作量核算' in filename and '__MACOSX' not in folderName:
            # print('1'*100)
            print(filename)
            # print('2'*100)
            return True
    return False

dst_list = []
for folderName, subfolders, filenames in os.walk('E:\\cosmic 评审\\2020年第一季度专家评审打分表.xlsx等文件\\专家评审材料'):

    if file_search(folderName, filenames):
        # print(filenames)
        print('The current folder is ' + folderName)
        dst = folderName.split('\\')[-1]
        print(dst)
        dst = 'E:\\organized_file\\' + dst
        print(dst)
        if dst not in dst_list:
            dst_list.append(dst)
            try:
                shutil.copytree(folderName, dst)
            except Exception as e:
                print(e)