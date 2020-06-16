#-*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:03
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : read_attachment9.py

import pandas as pd
import os, sys

pd.set_option('display.max_columns',20) #给最大列设置为10列

pd.set_option('display.max_rows',10)#设置最大可见10行

cosmic_cols = ['batch', 'project', 'requirementNO', 'requirement_name', 'requirement_detail', 'advocator',
                                      'days_spent', 'coding_requirement']

noncosmic_cols = ['batch', 'project', 'requirementNO', 'requirement_name', 'requirement_detail', 'work_cat',
                                          'work_name', 'work_detail', 'days_spent']

cosmic_info = pd.DataFrame(columns=cosmic_cols)

# 非cosmic的 人天、 工作类型、工作名称、工作内容详细描述加进去
noncosmic_info = pd.DataFrame(columns=noncosmic_cols)


def get_cosmic_info(cosmic_sheet):
    temp_info = pd.DataFrame(columns=cosmic_cols)

    part1 = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip() == '需求序号'].reset_index(drop=True)
    part1.dropna(inplace=True, axis=1)  ###去重空字段
    requirement_count = part1.shape[0]
    if part1.shape[1] == 4 and not (part1.iloc(axis=1)[1].str.strip() == '').any():
        temp_info['requirementNO'] = part1.iloc[:, 1]
    else:
        return False, 'Error: 请检查需求序号、需求名称格式'

    # 找到需求名称列且需求名称不能为空
    b1 = (part1.iloc[:, 2].str.strip() == '需求名称').all()
    b2 = (part1.iloc(axis=1)[3].str.strip() == '').any()
    if b1 and not b2:
        temp_info['requirement_name'] = part1.iloc[:, 3]
    else:
        return False, 'Error: 请检查需求名称是否存在'

    part2 = cosmic_sheet[cosmic_sheet.iloc(axis=1)[0].str.strip() == '需求提出人'].reset_index(drop=True)
    part2.dropna(inplace=True, axis=1)
    b3 = (part2.iloc[:, 1].str.strip() == '').any()
    if part2.shape[0] == requirement_count and part2.shape[1] == 6 and not b3:
        temp_info['advocator'] = part2.iloc[:, 1]
    else:
        return False, 'Error: 请检查需求提出人有关行（是否满足“需求提出人、实际工作量（人天）、需求预估工作量（人天）”格式）'

    # 找到'实际工作量（人天）'的列
    b4 = (part2.iloc[:, 2].str.strip().isin(['实际工作量（人天）'])).all()
    # 检查'实际工作量（人天）'是否存在空值
    b5 = (part2.iloc[:, 3] == '').any()
    try:
        part2.iloc(axis=1)[3] = part2.iloc(axis=1)[3].astype(float)
    except Exception as e:
        return False, 'Error: 请检查实际工作量是否为数值'
    if b4 and not b5:
        temp_info['days_spent'] = part2.iloc[:, 3]
    else:
        return False, 'Error: 提交的需求缺少实际工作量'
    part3 = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip() == '需求描述'].reset_index(drop=True)
    part3.iloc(axis=1)[2].fillna('', inplace=True)
    part3.dropna(inplace=True, axis=1)

    if part3.shape[0] == requirement_count:
        temp_info['requirement_detail'] = part3.iloc[:, 1]
    else:
        return False, 'Error: 提交的需求缺少需求详情'

    coding_selection = ['代码', '代码开发', '数据脚本']
    coding_requirement_indices = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip().isin(coding_selection)].index

    coding_requirement_indices = coding_requirement_indices + 2
    coding_requirement = cosmic_sheet.iloc[coding_requirement_indices, :]

    coding_requirement.iloc[:, 3].fillna('', inplace=True)
    coding_requirement.dropna(axis=1,inplace=True)
    coding_requirement.reset_index(drop=True, inplace=True)

    # 找到代码的功能点描述
    if coding_requirement.shape[0] == requirement_count and (coding_requirement.iloc(axis=1)[0].str.contains('功能')).all():
        temp_info['coding_requirement'] = coding_requirement.iloc[:, 1]
    else:
        return False, 'Error: 请检查代码开发是否满足“投入人员-功能点数量-功能名称列表”的描述格式'

    project_info = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip() == '所属系统/项目']
    project_info.dropna(inplace=True, axis=1)
    if project_info.iloc[0, 1].strip() != '':
        temp_info['project'] = project_info.iloc[0, 1]
    else:
        return False, 'Error: 请把“所属系统/项目”填写在文件头部'
    temp_info['batch'] = '2020Q1'
    # print(temp_info)
    return True, temp_info


def get_noncosmic_info():
    return

path = 'E:\\cosmic 评审\\2020年第一季度专家评审打分表.xlsx等文件\\专家评审材料'
file_count = 0
have_read = 0
for folderName, subfolders, filenames in os.walk(path):
    for filename in filenames:
        # 避免重复读取苹果系统格式的文件
        if '工作量核算' in filename and '__MACOSX' not in folderName:
            file_path = folderName + '\\' + filename
            # print(file_path)
            file_count += 1
            try:
                all = pd.read_excel(file_path, sheet_name=None, header=None)
                s2_key = list(all.keys())[1]
                s2 = all[s2_key]

                if len(all.keys()) >= 4 and (list(all.keys())[3]).contains('非COSMIC'):
                    s4 = all[s4_key]
                else:
                    print(folderName)

            except Exception as e:
                print(e)
                print('请检查文件：' + file_path)
                continue

            FLAG, cosmic_result = get_cosmic_info(s2)
            if FLAG:
                cosmic_info = cosmic_info.append(cosmic_result, ignore_index=True)
                have_read += 1
            else:
                print('读取cosmic列表出错')
                print(cosmic_result)
                print(file_path)
print('检索到文件' + str(file_count))
print('成功读取' + str(have_read))
print(cosmic_info)



# file_path = 'E:\\cosmic 评审\\2020年第一季度专家评审打分表.xlsx等文件\\专家评审材料\\2020-1-35-ERP系统维护\\附件9：工作量核算表（结算）-项目序号35.xls'
# all = pd.read_excel(file_path, sheet_name=None, header=None)
# s2_key = list(all.keys())[1]
# s2 = all[s2_key]
# FLAG, cosmic_result = get_cosmic_info(s2)
# print(FLAG, cosmic_result)





