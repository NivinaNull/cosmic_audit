#-*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:03
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : read_attachment9.py

import numpy as np, pandas as pd
import os, sys, platform
import copy, re
import math
import datetime
import logging
from collections import Counter
from pathlib import Path
import pickle

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
                int_nu = int(re_no.pop())
                return int_nu
    return None


def format2(item):
    if item:
        re_no = re.findall(r"\d+", item)
        re_no = [x for x in re_no if x not in year_list]
        if len(re_no) >= 1:
            for n in re_no:
                if n == item:
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
    part1 = cosmic_sheet[cosmic_sheet.iloc[:, 0].astype(str).str.strip() == '需求序号'].reset_index()
    index_stride = list(part1['index'])
    index_stride.append(cosmic_sheet.shape[0])
    # temp_info['start'] = index_stride[:-1]
    # temp_info['end'] = index_stride[1:]
    index_stride = [{'start':index_stride[i], 'end':index_stride[i + 1]} for i in range(len(index_stride) - 1)]
    temp_info['index_stride'] = index_stride
    part1 = part1.drop(columns='index').dropna(axis=1)  ###去重空字段
    if part1.empty:
        return False, 'ERROR: 《表2 需求开发工作量核算表》没有找到需求序号'
    requirement_count = part1.shape[0]

    if part1.shape[1] > 4:
        part1 = part1.T.drop_duplicates().T
    if part1.shape[1] == 4 and not (part1.iloc(axis=1)[1].astype(str).str.strip() == '').any():
        temp_info['requirementNO'] = part1.iloc[:, 1].astype(str)
        temp_info['requirementNO'] = temp_info['requirementNO'].map(format1)
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
    part2 = cosmic_sheet[cosmic_sheet.iloc(axis=1)[0].str.strip() == '需求提出人'].reset_index()
    if part2.empty:
        print('WARNING: 《表2 需求开发工作量核算表》 没有找到需求提出人相关行' + '\n', str(file_path))
        temp_info['advocator'] = '无'
        temp_info['days_spent'] = -1
    else:
        part2.drop(columns=[i for i in part2.columns if part2[i].isna().all()], inplace=True)
        if part2.shape[0] == requirement_count and part2.shape[1] >= 5:
            temp_info['advocator'] = part2.iloc[:, 2].fillna('无').astype(str)
            no_advocator = list(temp_info[temp_info.advocator == '无'].requirementNO)
            if len(no_advocator) > 0:
                print('WARNING: 《表2 需求开发工作量核算表》 <需求提出人> 存在漏填\t' + '需求编号为：' + str(no_advocator) + '\n', str(file_path))

            if (part2.iloc[:, 3].astype(str).str.strip().map(lambda x: True if '实际工作量' in x else False)).all():
                # print('part2.iloc[:, 4]_before ' * 8 + '\n', list(part2.iloc[:, 4]))
                part2.iloc[:, 4].fillna(-1, inplace=True)
                temp_info['days_spent'] = part2.iloc(axis=1)[4].astype(str).map(lambda x: re.findall("\d+", x)[0] if (len(re.findall("\d+", x)) == 1) else -1).astype(float)
                # print('part2.iloc[:, 4]_after ' * 8 + '\n', list(temp_info['days_spent']))
                located_nan2 = list(temp_info[temp_info.days_spent == -1]['requirementNO'].values)
                if len(located_nan2) > 0:
                    print('WARNING: 《表2 需求开发工作量核算表》 <实际工作量> 存在漏填\t', '需求序号为：' + str(located_nan2) + '\n', str(file_path))
            else:
                temp_info['days_spent'] = -1
                print('WARNING: 《表2 需求开发工作量核算表》 无法定位 <实际工作量> 相关信息' + '\n', str(file_path))
        else:
            print('WARNING: 《表2 需求开发工作量核算表》需求提出人相关行\n' +
                  '1.需求名称 相关行行数必须与需求序号条数保持一致'
                  '2.是否满足<需求提出人、实际工作量（人天）、需求预估工作量（人天）> 格式' + '\n',
                  str(file_path))
            temp_info['advocator'] = '无'
            temp_info['days_spent'] = -1


    part3 = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip() == '需求描述'].reset_index(drop=True)
    if part3.empty:
        return False, 'ERROR: 《表2 需求开发工作量核算表》 没有找到需求描述'
    part3.iloc(axis=1)[2].fillna('', inplace=True)
    part3.dropna(inplace=True, axis=1)

    if part3.shape[0] == requirement_count:
        temp_info['requirement_detail'] = part3.iloc[:, 1]
    else:
        return False, 'ERROR: 《表2 需求开发工作量核算表》提交的需求缺少需求详情'

    coding_selection = ['代码', '代码开发', '数据脚本', '数据配置']
    coding_requirement_indices = cosmic_sheet[cosmic_sheet.iloc[:, 0].str.strip().isin(coding_selection)].index
    # print('coding_requirement_indices_before ' * 10 + '\n', coding_requirement_indices)
    # print([coding_requirement_indices[i] for i in range(1, len(coding_requirement_indices))
    #                                                                      if ((coding_requirement_indices[i] - coding_requirement_indices[i-1]) > 1)])
    # index_checked = [coding_requirement_indices[0]]
    if coding_requirement_indices.empty:
        return False, 'ERROR: 该项目没有代码开发信息'
    coding_requirement_indices = [coding_requirement_indices[0]] + \
                                 [coding_requirement_indices[i] for i in range(1, len(coding_requirement_indices))if ((coding_requirement_indices[i] - coding_requirement_indices[i-1]) > 1)]

    coding_requirement_indices.extend([(i+1) for i in coding_requirement_indices] + [(i+2) for i in coding_requirement_indices])
    coding_requirement = cosmic_sheet.iloc[coding_requirement_indices, :]


    # 找到代码的功能点描述
    # print('coding_requirement_before '* 5 + '\n', coding_requirement)
    coding_requirement.drop(columns=[i for i in coding_requirement.columns if coding_requirement[i].isna().all()], inplace=True)
    # print('coding_requirement_after ' * 5 + '\n', coding_requirement, '\n', coding_requirement.columns)
    # print('contrast ' * 20 + '\n', coding_requirement.iloc(axis=1)[1], '\n', coding_requirement.iloc(axis=1)[2])

    if len(coding_requirement.columns) < 2:
        temp_info['coding_requirement'] = np.nan
    else:
        coding_requirement = coding_requirement.loc[(coding_requirement.iloc(axis=1)[1].astype(str).str.contains('功能'))
                                                    & (coding_requirement.iloc(axis=1)[1].astype(str).str.contains('列表'))]

        # print('coding_requirement ' * 10 + '\n', coding_requirement)
        # print('coding_requirement.shape[0] ' * 10 + '\n', coding_requirement.shape[0], '\n', list(coding_requirement.index))
        if coding_requirement.shape[0] != len(coding_requirement_indices) / 3:
            print('功能点名称列表 doesnt match 代码开发' + '\n', str(file_path), '\n', '代码开发->功能点名称 相关行数：',
                  coding_requirement.shape[0], '代码开发相关行数：', int(len(coding_requirement_indices) / 3))
        coding_requirement = coding_requirement.iloc(axis=1)[1:3]
        coding_requirement.iloc(axis=1)[1].fillna('无', inplace=True)
        # coding_requirement = coding_requirement.dropna(axis=1).astype(str)
        # print(coding_requirement)
        # print('len(coding_requirement.columns) ' * 8 + '\n', len(coding_requirement.columns))
        coding_requirement_list = []
        # if not coding_requirement.empty and len(coding_requirement.columns) == 2:
        if not coding_requirement.empty:
            coding_requirement.columns = ['name', 'detail']
            coding_requirement.reset_index(inplace=True)
            for r in temp_info['index_stride']:
                # print('coding_requirement[coding_requirement[\'index\'].between(r[\'start\'], r[\'end\'])]\n', coding_requirement[coding_requirement['index'].between(r['start'], r['end'])])
                value = coding_requirement[coding_requirement['index'].between(r['start'], r['end'])].detail.values
                # print(value)
                if value and (str(value[0]).strip() != '无') and (str(value[0]).strip() != '不涉及'):
                    # print('value_here ' * 10, value)
                    coding_requirement_list.extend(value)
                else:
                    # 没有找到<代码开发>相关行
                    coding_requirement_list.append(np.nan)
        else:
            coding_requirement_list = [np.nan] * requirement_count
        temp_info['coding_requirement'] = coding_requirement_list

    if temp_info.coding_requirement.hasnans:
        suspicious_req = temp_info.requirementNO[temp_info.coding_requirement.isna()]
        print('WARNING -- 发现没有<代码描述>的cosmic需求，序号：',suspicious_req.values, '\n',  str(file_path))

    temp_info.drop(columns=['index_stride'], inplace=True)
    temp_info['batch'] = '2020Q1'  ### current_batch
    # print('temp_info_last '* 5 + '\n', temp_info)

    return True, temp_info


# temp_noncosmic_cols = ['requirementNO', 'requirement_name', 'work_cat', 'work_name', 'work_detail', 'days_spent']
def get_noncosmic_info(noncosmic_sheet):
    temp_info = pd.DataFrame(columns=temp_noncosmic_cols)
    noncosmic_result = pd.DataFrame()
    if (noncosmic_sheet.iloc[:, 0] == '项目序号').any():
        start = (noncosmic_sheet[noncosmic_sheet.iloc[:, 0] == '项目序号']).index.values[0]
        noncosmic_sheet = noncosmic_sheet[start:]
        info_count = 0
        for i in range(noncosmic_sheet.shape[1]):
            # 找到文件非cosmic信息的有效开始部分
            if str(noncosmic_sheet.iloc[0, i]).strip() == '需求序号':
                temp_info['requirementNO'] = noncosmic_sheet.iloc[1:, i].astype(str)
                info_count += 1
            elif str(noncosmic_sheet.iloc[0, i]).strip() == '需求名称':
                temp_info['requirement_name'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '非COSMIC人天' in str(noncosmic_sheet.iloc[0, i]):
                temp_info['days_spent'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '工作类型' in str(noncosmic_sheet.iloc[0, i]):
                temp_info['work_cat'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '工作名称' in str(noncosmic_sheet.iloc[0, i]):
                temp_info['work_name'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
            elif '工作内容详细描述' in str(noncosmic_sheet.iloc[0, i]):
                temp_info['work_detail'] = noncosmic_sheet.iloc[1:, i]
                info_count += 1
        if info_count != 6:
            return False, 'ERROR: 请检查<表4 非COSMIC评估工作量填报表>的【需求序号、需求名称、非COSMIC人天、工作类型、工作名称、工作内容详细描述】是否完整'

        # 判断字段内容是否缺失，处理字段格式，需求序号统一处理成int，与cosmic部分保持一致

        temp_info.reset_index(drop=True, inplace=True)
        temp_info['requirementNO'] = temp_info['requirementNO'].map(format1)

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
            if str(sketch.iloc[r, c]).strip() == '项目名称':
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


def check_nonfiles(noncosmic_result):
    for f in filenames:
        if ('非COSMIC' in f or '非COSMIC'.lower() in f) and not f.startswith('.') and ('__MACOSX' not in folderName):
            local_path = Path(folderName) / f
            try:
                sheet = pd.read_excel(local_path, sheet_name=None, header=None)
                key = list(sheet.keys())[0]
                Another_chance, noncosmic_result = read_noncosmic_sheet(sheet, key)
                return Another_chance, noncosmic_result
            except Exception as error:
                print('ERROR:读取文件出错——请检查文件： ' + local_path + '\n', error)
    return False, noncosmic_result


if platform.system() == 'Windows':
    path = Path('C:\\ChinaMobile\\cosmic_files\\专家评审材料_2020Q3')
    # path = Path('C:\\ChinaMobile\\cosmic_files\\error_compound')
    # path = Path('C:\\ChinaMobile\\cosmic_files\\专家评审材料_2020Q3\\2020-3-206-2019年中国移动内蒙古公司BOSS系统软件技术服务框架合同')
else:
    path = Path('./cosmic_files')

file_count = 0
sketch_have_read = 0
cosmic_have_read = 0
noncosmic_have_read = 0
for folderName, subfolders, filenames in os.walk(path):
    # SEPARATE_NONCOSMIC = 0
    # # 检校非COSMIC工作量核算文件是否单独独立
    # print('folderName : ', folderName)
    # print('subfolders_before : ', subfolders)
    # print('filenames : ', filenames)

    temp_copy = copy.deepcopy(subfolders)
    for sf in temp_copy:
        if sf.startswith('.'):
            subfolders.remove(sf)
    # print('subfolders_after : ', subfolders)
    for filename in filenames:
        # 避免重复读取苹果系统格式的文件
        if ('工作量核算' in filename) and not filename.startswith('.') and ('__MACOSX' not in folderName):
            # file_name_split = re.split('项目序号|\.|_|' + current_year, filename)
            file_path = Path(folderName) / filename
            filename = re.sub("[\s+\.\!\/\:_,$%^*(+\"\']+|[+——！，。？、：~@#￥%……&*（）]+", "", filename)
            projectNo = -1
            s = re.search('项目序号\d+', filename)
            if s:
                matched = s.group(0)
                projectNo = re.findall("\d+", matched)
                if len(projectNo) == 1:
                    projectNo = projectNo[0]
                else:
                    print('ERROR:请检查文件名称是否满足《附件9：工作量核算表（结算）-项目序号xxx.xls》的文件格式\n' + str(file_path))
                    continue
            else:
                print('ERROR:请检查文件名称是否满足《附件9：工作量核算表（结算）-项目序号xxx.xls》的文件格式\n' + str(file_path))
                continue
            s1_key, s2_key, s4_key = '', '', ''
            try:
                all = pd.read_excel(file_path, sheet_name=None, header=None)
            except Exception as error:
                print('ERROR:读取文件出错——请检查文件：' + str(file_path) + '\n', error)
                continue

            file_count += 1
            if len(list(all.keys())) >= 4 and '非COSMIC' in list(all.keys())[3]:
                s1_key = list(all.keys())[0]
                s2_key = list(all.keys())[1]
                s4_key = list(all.keys())[3]
            elif len(list(all.keys())) >= 2:
                s1_key = list(all.keys())[0]
                s2_key = list(all.keys())[1]
            else:
                print('ERROR:请检查文件的《表2 需求开发工作量核算表》是否存在\n' + str(file_path))
                continue

            # sheet1部分处理
            s1 = all[s1_key]
            FLAG, requirements_result, project_name = get_requirements(s1)
            if FLAG:
                sketch_have_read += 1
            else:
                print('requirements_result_fail_to_read'* 5 + '\n', project_name, '\n', requirements_result, '\n', str(file_path))
                continue

            # sheet2、sheet4部分
            FLAG1, cosmic_result = read_cosmic_sheet(all, s2_key)
            if not FLAG1:
                print(cosmic_result, '\n', str(file_path))
            else:
                ####################################################current_batch###################################
                cosmic_result['batch'] = '2020Q1'
                cosmic_result['project_name'] = project_name
                cosmic_result['projectNo'] = projectNo
                cosmic_result.drop_duplicates(subset=['requirement_name'], inplace=True)
                cosmic_info = cosmic_info.append(cosmic_result, ignore_index=True)
                cosmic_have_read += 1

            FLAG2, noncosmic_result = read_noncosmic_sheet(all, s4_key)
            if not FLAG2:
                if check_nonfiles(noncosmic_result)[0]:
                    FLAG2, noncosmic_result = check_nonfiles(noncosmic_result)
                    noncosmic_result['batch'] = '2020Q1'
                    noncosmic_result['projectNo'] = projectNo
                    noncosmic_result['project_name'] = project_name
                    noncosmic_result.drop_duplicates(subset=['requirement_name'], inplace=True)
                    noncosmic_info = noncosmic_info.append(noncosmic_result, ignore_index=True)
                    noncosmic_have_read += 1
                else:
                    print(noncosmic_result, '\n', str(file_path))

            else:
                ####################################################current_batch###################################
                noncosmic_result['batch'] = '2020Q1'
                noncosmic_result['projectNo'] = projectNo
                noncosmic_result['project_name'] = project_name
                noncosmic_result.drop_duplicates(subset=['requirement_name'], inplace=True)
                noncosmic_info = noncosmic_info.append(noncosmic_result, ignore_index=True)
                noncosmic_have_read += 1


            # 判断是否在表2表4的需求是否和表1的需求一致
            set_union = set()
            if (FLAG1 and FLAG2):
                set_union = set(cosmic_result.requirementNO) | set(noncosmic_result.requirementNO)
            elif FLAG1:
                set_union = set(cosmic_result.requirementNO)

            elif FLAG2:
                set_union = set(noncosmic_result.requirementNO)

            if set(requirements_result.requirementNO) == set_union:
                continue
            else:
                print('WARNING:sheet2与sheet4的需求总集合与sheet1的需求集合不符\n',
                      'sheet2与sheet4的需求总集合：\n', set_union, '\n',
                      'sheet1需求集合：\n',set(requirements_result.requirementNO), '\n',
                      str(file_path))


print('检索到文件：' + str(file_count))
print('成功读取工作量核算汇总表的文件：' + str(sketch_have_read))
print('成功读取cosmic信息的文件：' + str(cosmic_have_read))
print('成功读取非cosmic信息的文件' + str(noncosmic_have_read))
print('over_ here ' * 8 + '\n', cosmic_info)
print(noncosmic_info)

#直接覆盖
# cosmic_info.to_pickle('./data/cosmic_info_new.pkl')
# noncosmic_info.to_pickle('./data/noncosmic_info_new.pkl')

# cosmic_info.to_csv('./data/cosmic_info.csv', encoding="utf_8_sig", index=False)
# noncosmic_info.to_csv('./data/noncosmic_info.csv', encoding="utf_8_sig", index=False)

#追加信息
# with open('./data/cosmic_info.pkl', mode='a') as fd1:
#     pickle.dump(cosmic_info, fd1)
# with open('./data/noncosmic_info.pkl', mode='a') as fd2:
#     pickle.dump(noncosmic_info, fd2)
