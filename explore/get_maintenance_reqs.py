#-*- coding: utf-8 -*-
# @Time    : 14:27 
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : get_maintenance_reqs
from pathlib import Path
import os
import math
import pandas as pd

pd.set_option('display.max_columns',20) #给最大列设置为10列
pd.set_option('display.max_rows',10)#设置最大可见10行


class GetMaintenanceReqs:

    column_list = ['type', 'projectNO', 'project_name', 'batch', 'requirementNO', 'requirement_name',
                   'requirement_detail', 'advocator', 'days_spent', 'phase_detail']
    reflection = {
        'batch': '希望完成日期',
        'requirementNO': '需求编号',
        'requirement_name': '需求名称',
        'requirement_detail': '需求描述',
        'advocator': '提交人',
        'days_spent': '实际工作量',  ####实际工作量 原文件单位为‘人时’，在此需要转换单位为‘人天’
        'phase_detail': '任务完成说明'
    }
    result_data = pd.DataFrame(columns=column_list)

    def __init__(self, source_path, target_path):
        self.source = source_path
        self.target = target_path

    def _read_file(self, file_path):

        def to_std(item):
            if len(item) == 8:
                s = 'Q' + str(math.ceil(int(item[4:6]) / 3))
                return (item[:4] + s)
            else:
                return None

        pn = file_path.stem
        file_name = file_path.name
        one_piece = pd.DataFrame(columns=self.column_list)
        file_df = pd.read_excel(file_path, sheet_name=None)
        file_df = file_df[list(file_df.keys())[0]]
        file_df = file_df[~(file_df['需求编号'].isna())]
        success_count = 0
        for key, value in self.reflection.items():
            if value in file_df.columns:
                one_piece[key] = file_df[value]
                success_count += 1
        if success_count != success_count:
            print('WARNING：未读入文件完整信息', '\n', file_path)
        if not one_piece.empty:
            one_piece['type'] = '运维'
            one_piece['projectNO'] = '-1'
            one_piece['project_name'] = pn
            one_piece['file_trail'] = file_name
            # 将单位小时转化为工作日（一工作日8小时）
            one_piece['days_spent'] = one_piece['days_spent'].astype(int) / 8

            try:
                one_piece.batch = one_piece.batch.astype(int).astype(str).map(to_std)
            except Exception as e:
                one_piece.batch = None
                print('WARNING：希望完成日期 未成功转成 batch')
            return True, one_piece
        else:
            return False, None

    def read_files(self):
        for foldername, subfolders, filenames in os.walk(self.source):
            for fn in filenames:
                if fn.endswith('.xlsx') or fn.endswith('.xls'):
                    result = self._read_file(Path(foldername, fn))
                    if result[0]:
                        self.result_data = self.result_data.append(result[1], ignore_index=True)
                    else:
                        print('ERROR：文件读入失败', '\n', str(Path(foldername, fn)))
        return self.result_data

    def duplicate_check(self):
        if_duplicated = self.result_data.duplicated(subset=['project_name', 'batch', 'requirementNO', 'requirement_name'], keep=False)
        if if_duplicated.any():
            duplicated_info = self.result_data[if_duplicated]['project_name', 'batch', 'requirementNO', 'requirement_name']
            print('WARNING：本批次运营需求存在相同需求，如下所示：\n', duplicated_info)
        else:
            print('本批次运营需求未检测到相同需求')

    def persist(self):
        out_file = self.target / 'maintenance_req.pkl'
        if out_file.is_file():
            print('pickle文件已存在')
            his = pd.read_pickle(out_file)
            his = his.append(self.result_data, ignore_index=True)
            # his.drop_duplicates(inplace=True)
            his.to_pickle(out_file)
        else:
            self.result_data.to_pickle(out_file)


if __name__ == '__main__':

    source = Path('C:\\ChinaMobile\\maintenance_requirements')
    target = Path('./data')
    get_reqs = GetMaintenanceReqs(source, target)
    maintenance_reqs = get_reqs.read_files()
    print(maintenance_reqs)
    # 预检查是否存在需求填写重复
    get_reqs.duplicate_check()
    ####### 数据持久化，运行多次会导致将该次读取的数据多次追加，请确认同一批次的需求只持久化一次
    get_reqs.persist()