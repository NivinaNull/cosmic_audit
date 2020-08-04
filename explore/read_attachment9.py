#-*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:03
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : read_attachment9.py

import numpy as np, pandas as pd
import os, sys
import re
import math
import datetime
from collections import Counter

current_year = str(datetime.datetime.now().year)
current_quarter = str(int((datetime.datetime.now().month - 1) / 3 + 1))
current_batch = current_year + 'Q' + current_quarter


pd.set_option('display.max_columns',20) #给最大列设置为10列
pd.set_option('display.max_rows',10)#设置最大可见10行

cosmic_cols = ['batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'requirement_detail', 'advocator',
                                      'days_spent', 'coding_requirement']

noncosmic_cols = ['batch', 'project_name', 'projectNo', 'requirementNO', 'requirement_name', 'work_cat', 'work_name',
                                    'work_detail', 'days_spent']
temp_noncosmic_cols = ['requirementNO', 'requirement_name', 'work_cat', 'work_name', 'work_detail', 'days_spent']

cosmic_info = pd.DataFrame(columns=cosmic_cols)

# 非cosmic的 人天、 工作类型、工作名称、工作内容详细描述加进去
noncosmic_info = pd.DataFrame(columns=noncosmic_cols)


year_list = [str(i + datetime.datetime.now().year) for i in range(-5, 6)]
def format1(item):
    # re_no = re.split('年|需求|序号|号|编号|\d季度|季度\d|' + current_year, item)
    if item:
        # item = ''.join(re.split('\d季度|季度\d|' + '|'.join(year_list), item))
        item = ''.join(re.split('[1-4]Q|Q[1-4]|[1-4]季度|季度[1-4]', item))
        re_no = re.findall("\d+", item)
        if len(re_no) >= 1:
            re_no = set(re_no) - set(year_list)
            if len(re_no) == 1:
                return int(re_no.pop())
    return None


def format2(item):
    if item:
        re_no = re.findall(r"\d+", item)
        re_no = [x for x in re_no if x not in year_list]
        if len(re_no) >= 1:
            for n in re_no:
                if n==item:
                    return None
                m = re.search('^(.*需求.*?|.*序号.*?|.*编号.*?|)' + n, item)
                if m:
                    matched = m.group(0)
                    if ('合计' in matched) or (('共'+ n  in item) or (n + '个' in item) or (n + '需求' in item) and len(item) <= 8):
                        return None
                    name = ''
                    try:
                        name = re.split(':|：|' + matched, item)
                    except Exception as e:
                        return None
                    name = ''.join(name)
                    if not name:
                        return None
                    return int(n), name
    return None


def get_cosmic_info(cosmic_sheet):
    temp_info = pd.DataFrame(columns=cosmic_cols)
    part1 = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip() == '需求序号'].reset_index(drop=True)
    part1.dropna(inplace=True, axis=1)  ###去重空字段
    requirement_count = part1.shape[0]
    # 需求序号/需求序号_value/需求名称/需求名称_value的格式，并且需求序号_value不能有空值
    if part1.shape[1] == 4 and not (part1.iloc(axis=1)[1].str.strip() == '').any():
        temp_info['requirementNO'] = part1.iloc[:, 1].astype(str)
        # print('1' * 30 + '\n', list(temp_info['requirementNO']))
        temp_info['requirementNO'] = temp_info['requirementNO'].map(format1)
        # print('2' * 30 + '\n', list(temp_info['requirementNO']))
        # print('temp_info信息' * 100 + '\n', temp_info['requirementNO'])
        if temp_info['requirementNO'].isnull().any():
            return False, 'ERROR: 请检查《表2 需求开发工作量核算表》需求序号填写信息是否满足：xxxx年需求序号xxxx 的格式'
    else:
        return False, 'ERROR: 请检查《表2 需求开发工作量核算表》需求序号、需求名称格式'

    # 找到需求名称列且需求名称不能为空
    b1 = (part1.iloc[:, 2].str.strip() == '需求名称').all()
    b2 = (part1.iloc(axis=1)[3].str.strip() == '').any()
    if b1 and not b2:
        temp_info['requirement_name'] = part1.iloc[:, 3]
    else:
        return False, 'ERROR: 请检查《表2 需求开发工作量核算表》需求名称是否存在'

    part2 = cosmic_sheet[cosmic_sheet.iloc(axis=1)[0].str.strip() == '需求提出人'].reset_index(drop=True)
    # print('part2' * 30 + '\n', list(part2.columns))
    part2.dropna(inplace=True, axis=1)
    b3 = (part2.iloc[:, 1].str.strip() == '').any()
    if part2.shape[0] == requirement_count and part2.shape[1] == 6 and not b3:
        temp_info['advocator'] = part2.iloc[:, 1]
    else:
        return False, 'ERROR: 请检查《表2 需求开发工作量核算表》需求提出人有关行\n' \
                      '1.是否满足<需求提出人、实际工作量（人天）、需求预估工作量（人天）> 格式\n' \
                      '2.需求提出人、实际工作量（人天）、需求预估工作量（人天）是否存在漏填'

    # 找到'实际工作量（人天）'的列
    b4 = (part2.iloc[:, 2].str.strip().map(lambda x:True if '实际工作量' in x else False)).all()
    # 检查'实际工作量（人天）'是否存在空值
    b5 = (part2.iloc[:, 3] == '').any()
    try:
        part2.iloc(axis=1)[3] = part2.iloc(axis=1)[3].map(lambda x: x.split('人天')[0] if type(x)==str else x).astype(float)
    except Exception as e:
        return False, str(e) + 'ERROR: 请检查《表2 需求开发工作量核算表》实际工作量是否为数值'
    if b4 and not b5:
        temp_info['days_spent'] = part2.iloc[:, 3]
    else:
        return False, 'ERROR: 《表2 需求开发工作量核算表》提交的需求缺少实际工作量'
    part3 = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip() == '需求描述'].reset_index(drop=True)
    part3.iloc(axis=1)[2].fillna('', inplace=True)
    part3.dropna(inplace=True, axis=1)

    if part3.shape[0] == requirement_count:
        temp_info['requirement_detail'] = part3.iloc[:, 1]
    else:
        return False, 'ERROR: 《表2 需求开发工作量核算表》提交的需求缺少需求详情'

    coding_selection = ['代码', '代码开发', '数据脚本', '数据配置']
    coding_requirement_indices = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip().isin(coding_selection)].index

    if len(coding_requirement_indices) != requirement_count:
        return False, 'ERROR: 《表2 需求开发工作量核算表》中<代码开发>部分必须和需求个数一一对应（必须存在，可以不填）'
    coding_requirement_indices = coding_requirement_indices + 2
    coding_requirement = cosmic_sheet.iloc[coding_requirement_indices, :]
    # print('coding_requirement' * 10 + '\n', coding_requirement)
    coding_requirement.iloc[:, 3].fillna('', inplace=True)
    coding_requirement.dropna(axis=1,inplace=True)
    coding_requirement.reset_index(drop=True, inplace=True)
    # print('coding_requirement' * 10 + '\n', coding_requirement)

    # 找到代码的功能点描述

    coding_requirement_list = []
    for i in range(requirement_count):
        if '功能' in coding_requirement.iloc[i, 0]:
            coding_requirement_list.append(coding_requirement.iloc[i, 1])
        else:
            # coding_requirement.iloc[i, 0] == '无'
            coding_requirement_list.append('')
    temp_info.coding_requirement = coding_requirement_list
    # temp_info = temp_info[~(temp_info.coding_requirement == '')]
    temp_info['batch'] = '2020Q1'  ### current_batch
    # print('2' * 30 + '\n', temp_info)
    return True, temp_info


# temp_noncosmic_cols = ['requirementNO', 'requirement_name', 'work_cat', 'work_name', 'work_detail', 'days_spent']

def get_noncosmic_info(noncosmic_sheet):
    temp_info = pd.DataFrame(columns=temp_noncosmic_cols)
    noncosmic_result = pd.DataFrame()
    # print(noncosmic_sheet.shape)
    if (noncosmic_sheet.iloc[:, 0] == '项目序号').any():
        # print('index here' * 6, (noncosmic_sheet[noncosmic_sheet.iloc[:, 0] == '项目序号']).index)
        start = (noncosmic_sheet[noncosmic_sheet.iloc[:, 0] == '项目序号']).index.values[0]
        noncosmic_sheet = noncosmic_sheet[start:]
        # print(noncosmic_sheet)
        info_count = 0
        # print(noncosmic_sheet)
        for i in range(noncosmic_sheet.shape[1]):
            # 找到文件非cosmic信息的有效开始部分
            if noncosmic_sheet.iloc[0, i] == '需求序号':
                temp_info['requirementNO'] = noncosmic_sheet.iloc[1:, i].astype(str)
                info_count += 1
            elif noncosmic_sheet.iloc[0, i] == '需求名称':
                temp_info['requirement_name'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '非COSMIC人天' in noncosmic_sheet.iloc[0, i]:
                temp_info['days_spent'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '工作类型' in noncosmic_sheet.iloc[0, i]:
                temp_info['work_cat'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '工作名称' in noncosmic_sheet.iloc[0, i]:
                temp_info['work_name'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '工作内容详细描述' in noncosmic_sheet.iloc[0, i]:
                temp_info['work_detail'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
        if info_count != 6:
            return False, 'ERROR: 请检查<表4 非COSMIC评估工作量填报表>的【需求序号、需求名称、非COSMIC人天、工作类型、工作名称、工作内容详细描述】是否完整'

        # 判断字段内容是否缺失，处理字段格式，需求序号统一处理成int，与cosmic部分保持一致

        temp_info.reset_index(drop=True, inplace=True)
        # print('3' *30 + '\n', list(temp_info['requirementNO']))
        temp_info['requirementNO'] = temp_info['requirementNO'].map(format1)
        # print('4' *30 + '\n', list(temp_info['requirementNO']))

        # 只取非cosmic信息的有效部分，过滤掉无效的尾部信息
        temp_info = temp_info[temp_info['work_cat'].notnull() & temp_info['work_name'].notnull() & temp_info['work_detail'].notnull()]
        if temp_info.empty:
            return False, 'ERROR:请检查<表4 非COSMIC评估工作量填报表>中【工作种类，工作名称，工作细节】能否被正确识别或为空'
        temp_info.update(temp_info[['requirementNO', 'requirement_name']].fillna(method='ffill'))
        # 检查requirementNO是否仍旧存在空值
        if temp_info['requirementNO'].isnull().any():
            return False, 'ERROR: 请检查<表4 非COSMIC评估工作量填报表>中需求序号是否存在漏填'
        temp_info['requirementNO'] = temp_info['requirementNO'].astype(int)
        temp_info['days_spent'] = temp_info['days_spent'].astype(float)
        # 防止需求序号漏填、错填 导致的需求丢失
        # print('start' * 20 + '\n')
        # print(list(temp_info['requirementNO']))
        if len(set(temp_info['requirementNO'].astype(str) + temp_info['requirement_name'])) != len(set(temp_info['requirementNO'])):
            return False, 'ERROR: 请检查<表4 非COSMIC评估工作量填报表>中的需求序号、需求名称是否存在漏填或错填'
        for requirementNO, part in temp_info.groupby(['requirementNO']):
            if len(set(part['days_spent'].dropna().to_list())) == 1:
                temp = pd.DataFrame()
                temp['requirementNO'] = [requirementNO]
                temp['requirement_name'] = [part['requirement_name'].iloc[0]]
                temp['days_spent'] = [set(part['days_spent'].dropna().to_list()).pop()]
                temp['work_cat'] = [part['work_cat'].to_list()]
                temp['work_name'] = [part['work_name'].to_list()]
                temp['work_detail'] = [part['work_detail'].to_list()]
                noncosmic_result = noncosmic_result.append(temp, ignore_index=True)
                noncosmic_result.requirementNO = noncosmic_result.requirementNO.astype(int)
            else:
                return False, 'ERROR: 请检查<表4 非COSMIC评估工作量填报表>中的 非COSMIC人天 填写是否正确(同一个需求只能对应一个)'
        return True, noncosmic_result
    else:
        return False, 'ERROR: 无法定位<表4 非COSMIC评估工作量填报表>中的项目序号'


def get_requirements(sketch):
    temp = pd.DataFrame()
    project_name_list = []
    project_name = ''
    for r in range(3):
        for c in range(sketch.shape[1]):
            if type(sketch.iloc[r, c]) == str and sketch.iloc[r, c].strip() == '项目名称':
                project_name_list = sketch.iloc[(r+1):, c].to_list()
            elif type(sketch.iloc[r, c]) == str and sketch.iloc[r, c].strip() == '需求名称':
                temp = sketch.iloc[(r+1):, c].astype(str)
    if temp.empty:
        return False, temp, 'ERROR, 定位不到<表1 工作量核算表 >中的需求名称'

    project_name_list = [x for x in project_name_list if type(x) == str and x.strip() != '合计' and x.strip() != '']
    if len(project_name_list) == 0:
        return False, temp, 'ERROR, 检查<表1 工作量核算表 >中的【项目名称】是否填写正确'
    project_name_list = re.split(':|：', project_name_list[0])
    if '项目序号' in project_name_list[0]:
        del(project_name_list[0])
        project_name = ''.join(project_name_list)
    else:
        project_name = ''.join(project_name_list)
    # print('5' * 30 + '\n', list(temp.values))
    temp = temp.map(format2).dropna()
    # print('6' * 30 + '\n', list(temp.values))
    # print([x[0] for x in temp.values])
    if temp.empty:
        return False, temp, 'ERROR, 请检查<表1 工作量核算表 >中的需求名称是否满足{需求序号XXX：需求名称}的命名方式'
    temp = pd.DataFrame(temp.values, columns=['origin'])
    temp['requirementNO'] = [x[0] for x in temp.origin.values]
    temp['requirement_name'] = [x[1] for x in temp.origin.values]
    temp.drop(columns=['origin'], inplace=True)
    return True, temp, project_name


def read_cosmic_sheet(all, s2_key):
    s2 = all[s2_key]
    if s2.empty:
        return False, 'ERROR:请检查文件的《表2 需求开发工作量核算表》信息'
    else:
        return get_cosmic_info(s2)


def read_noncosmic_sheet(all, s4_key):
    if s4_key:
        s4 = all[s4_key]
        if s4.empty:
            return False, 'ERROR:请检查文件的《表4 非COSMIC评估工作量填报表》信息'
        else:
            return get_noncosmic_info(s4)
    else:
        return False, 'ERROR:请检查文件的《表4 非COSMIC评估工作量填报表》是否存在'


path = 'D:\\Audit\\专家评审材料'
# path = 'D:\\Audit\\专家评审材料'
file_count = 0
sketch_have_read = 0
cosmic_have_read = 0
noncosmic_have_read = 0
for folderName, subfolders, filenames in os.walk(path):
    for filename in filenames:
        # 避免重复读取苹果系统格式的文件
        if '工作量核算' in filename and '__MACOSX' not in folderName:
            # file_name_split = re.split('项目序号|\.|_|' + current_year, filename)
            file_path = folderName + '\\' + filename
            filename = re.sub("[\s+\.\!\/\:_,$%^*(+\"\']+|[+——！，。？、：~@#￥%……&*（）]+", "", filename)
            projectNo = -1
            s = re.search('项目序号\d+', filename)
            if s:
                matched = s.group(0)
                projectNo = re.findall("\d+", matched)
                if len(projectNo) == 1:
                    projectNo = projectNo[0]
                else:
                    print('ERROR:请检查文件名称是否满足《附件9：工作量核算表（结算）-项目序号xxx.xls》的文件格式\n' + file_path)
                    continue
                # print('projectNo' * 20, projectNo, '\n', file_path)
            else:
                print('ERROR:请检查文件名称是否满足《附件9：工作量核算表（结算）-项目序号xxx.xls》的文件格式\n' + file_path)
                continue
            s1_key, s2_key, s4_key = '', '', ''
            try:
                all = pd.read_excel(file_path, sheet_name=None, header=None)
            except Exception as e:
                print('ERROR:读取文件出错——请检查文件：' + file_path)
                continue

            if len(list(all.keys())) >= 4 and '非COSMIC' in list(all.keys())[3]:
                s1_key = list(all.keys())[0]
                s2_key = list(all.keys())[1]
                s4_key = list(all.keys())[3]
            elif len(list(all.keys())) >= 2:
                s1_key = list(all.keys())[0]
                s2_key = list(all.keys())[1]
            else:
                print('ERROR:请检查文件的《表2 需求开发工作量核算表》是否存在\n' + file_path)
                continue
            file_count += 1

            # sheet1部分处理
            s1 = all[s1_key]
            FLAG, requirements_result, project_name = get_requirements(s1)
            if FLAG:
                sketch_have_read += 1
            else:
                print('requirements_result_fail_to_read'* 5 + '\n', project_name, '\n', requirements_result, '\n', file_path)
                continue

            # sheet2、sheet4部分
            cosmic_result, noncosmic_result = '', ''
            FLAG1, FLAG2 = False, False
            for i in range(2):
                if i == 0:
                    FLAG1, cosmic_result = read_cosmic_sheet(all, s2_key)
                    if not FLAG1:
                        print(cosmic_result, '\n', file_path)
                        continue
                    cosmic_result['batch'] = '2020Q1'   ### current_batch
                    cosmic_result['projectNo'] = projectNo
                    cosmic_result['project_name'] = project_name
                elif i == 1:
                    FLAG2, noncosmic_result = read_noncosmic_sheet(all, s4_key)
                    if not FLAG2:
                        print(noncosmic_result, '\n', file_path)
                        continue
                    noncosmic_result['batch'] = '2020Q1'  ###current_batch
                    noncosmic_result['projectNo'] = projectNo
                    noncosmic_result['project_name'] = project_name

            # 判断是否在表2表4的需求是否和表1的需求一致
            set_union = set()
            if (FLAG1 and FLAG2):
                # print(cosmic_result, '\n', noncosmic_result)
                set_union = set(cosmic_result.requirementNO) | set(noncosmic_result.requirementNO)
            elif FLAG1:
                set_union = set(cosmic_result.requirementNO)
            elif FLAG2:
                set_union = set(noncosmic_result.requirementNO)
            if set(requirements_result.requirementNO) == set_union:
                if FLAG1:
                    if not cosmic_result.empty:
                        cosmic_requirements_counter = dict(Counter(cosmic_result.requirement_name))
                        if set(cosmic_requirements_counter.values()) != {1}:
                            to_be_checked = '、'.join(
                                [key + '、次数：' + str(value) for key, value in cosmic_requirements_counter.items() if value > 1])
                            print('WARN:cosmic信息中存在相同需求名称：' + to_be_checked + '\n' + file_path)
                            continue
                    # cosmic_result.drop_duplicates(subset=['requirement_name'], inplace=True)
                    cosmic_info = cosmic_info.append(cosmic_result, ignore_index=True)
                    cosmic_have_read += 1
                if FLAG2:
                    if not noncosmic_result.empty:
                        noncosmic_requirements_counter = dict(Counter(noncosmic_result.requirement_name))
                        if set(noncosmic_requirements_counter.values()) != {1}:
                            to_be_checked = '、'.join(
                                [key + str(value) for key, value in noncosmic_requirements_counter.items() if value > 1])
                            print('Warning:noncosmic信息中存在相同需求名称' + to_be_checked)
                            continue
                    noncosmic_info = noncosmic_info.append(noncosmic_result, ignore_index=True)
                    noncosmic_have_read += 1
            else:
                print('ERROR:sheet2与sheet4的需求总集合与sheet1的需求集合不符\n',
                      'sheet2与sheet4的需求总集合：\n', set_union, '\n',
                      'sheet1需求集合：\n',set(requirements_result.requirementNO), '\n',
                      file_path)
                continue


print('检索到文件：' + str(file_count))
print('成功读取工作量核算汇总表的文件：' + str(sketch_have_read))
print('成功读取cosmic信息的文件：' + str(cosmic_have_read))
print('成功读取非cosmic信息的文件' + str(noncosmic_have_read))
print(cosmic_info)
print(noncosmic_info)

cosmic_info.to_pickle('./data/cosmic_info.pkl')
noncosmic_info.to_pickle('./data/noncosmic_info.pkl')
