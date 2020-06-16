#-*- coding: utf-8 -*-
# @Time    : 2020/6/5 15:48
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : explore1.py

import pandas as pd

pd.set_option('display.max_columns',20) #给最大列设置为10列
pd.set_option('display.max_rows',100)#设置最大可见100行


def get_cosmic_info(cosmic_sheet):
    temp_info = pd.DataFrame(columns=['batch', 'project', 'requirementNO', 'requirement_name', 'requirement_detail',
                                      'advocator', 'days_spent', 'coding_requirement'])
    requirement1 = cosmic_sheet[cosmic_sheet.iloc[:, 0] == '需求序号'].reset_index(drop=True)
    requirement1.dropna(inplace=True, axis=1)
    # print(requirement1)
    # print(requirement1.shape)
    temp_info['requirementNO'] = requirement1.iloc[:, 1]
    temp_info['requirement_name'] = requirement1.iloc[:, 3]

    requirement2 = cosmic_sheet[cosmic_sheet.iloc[:, 0] == '需求提出人'].reset_index(drop=True)
    requirement2.dropna(inplace=True, axis=1)
    # print(requirement2)
    # print(requirement2.shape)
    if requirement2.shape[0] == requirement1.shape[0]:
        temp_info['advocator'] = requirement2.iloc[:, 1]
        temp_info['days_spent'] = requirement2.iloc[:, 3]
        # print( requirement2.iloc[:, 1])
        # print(temp_info['advocator'])
        # print( requirement2.iloc[:, 3])
        # print(temp_info['days_spent'])
    else:
        # print('Error: 需求提出人个数与需求个数不一致')
        # print(requirement2.shape[0], requirement1.shape[0])
        return 'Error: 需求提出人个数与需求个数不一致'
    requirement3 = cosmic_sheet[cosmic_sheet.iloc[:, 0] == '需求描述'].reset_index(drop=True)
    requirement3.dropna(inplace=True, axis=1)
    # print(requirement3)
    if requirement3.shape[0] == requirement1.shape[0]:
        temp_info['requirement_detail'] = requirement3.iloc[:, 1]
    else:
        # print('Error: 需求描述个数与需求个数不一致')
        # print(requirement3.shape[0], requirement1.shape[0])
        return 'Error: 需求提出人个数与需求个数不一致'

    coding_selection = ['代码', '代码开发']
    coding_requirement_indices = cosmic_sheet[cosmic_sheet.iloc[:, 0].isin(coding_selection)].index
    coding_requirement_indices = [i + 2 for i in coding_requirement_indices]

    print(coding_requirement_indices)
    print(type(coding_requirement_indices))
    coding_requirement = cosmic_sheet.iloc[coding_requirement_indices, :]
    coding_requirement.dropna(axis=1,inplace=True)
    coding_requirement.reset_index(drop=True, inplace=True)
    # print(coding_requirement)
    if coding_requirement.shape[0] == requirement1.shape[0]:
        temp_info['coding_requirement'] = coding_requirement.iloc[:, 1]
    else:
        return 'Error: 需求提出人个数与代码需求个数不一致'

    project_info = cosmic_sheet[cosmic_sheet.iloc[:, 0] == '所属系统/项目']
    project_info.dropna(inplace=True, axis=1)
    print(project_info)
    temp_info['batch'] = '2020Q1'
    temp_info['project'] = project_info.iloc[0, 1]

    print(temp_info)


# file_path = "./data/附件7：需求序号001：COSMIC软件评估功能点拆分表.xlsx"
# ws_name = '功能点拆分表'
# cosmic_metric = pd.DataFrame(columns=['batch', 'project', 'requirementNO', 'requirement_name', 'requirement_detail', 'advocator',
                                      # 'days_spent', 'coding_requirement'])


file_path = 'E:\\organized_file\\专家评审材料\\附件9：工作量核算表（结算）-项目序号169.xls'
all = pd.read_excel(file_path, sheet_name=None, header=None)
s2_key = list(all.keys())[1]
s2 = all[s2_key].drop(index=0).reset_index(drop=True)
get_cosmic_info(s2)





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