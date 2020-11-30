#-*- coding: utf-8 -*-
# @Time    : 2020/7/20 14:00
# @Author  : Nivina
# @Email   : nivinanull@163.com
# @File    : upzip.py

import os
import zlib
import unrar
import shutil
import zipfile
import tarfile
from pathlib import Path
from time import sleep
from unrar import rarfile
import chardet
import copy
import re


class BaseTool(object):
    def __init__(self, path):
        self.path = path
        self.compress = [".tar.gz",".tar.bz2",".tar.bz",".tar.tgz",".tar",".tgz",".zip",".rar"]

    def iszip(self,  file):
        for z in self.compress:
            if file.endswith(z):
                return z

    def zip_to_path(self, file):
        for i in self.compress:
            file = file.replace(i,os.sep)
        # 解决win系统文件最大长度限制  以及包含'.'的特殊字符
        if os.name == 'nt':
            if not file.startswith('\\\\?\\'):
                file = file.replace('.', '_')
                file = '\\\\?\\' + file
        return file

    def error_record(self, info):
        error_output = Path('./run_error/error_record_test.txt')
        if not error_output.is_file():
            error_output.parent.mkdir(exist_ok=True)
            with error_output.open( "w", encoding='UTF-8') as r:
                r.write(info+"\n")
        else:
            with error_output.open("a", encoding='UTF-8') as r:
                r.write(info+"\n")

    def un_zip(self, src, dst):
        try:
            uz_path = self.zip_to_path(dst)
            # print('uz_path ' * 8 + '： ', uz_path)
            if not os.path.exists(uz_path):
                os.makedirs(uz_path)
            # print('zip_file.namelist() ' * 8 + '\n', zip_file.namelist())
            with zipfile.ZipFile(src, 'r') as f:
                print(len(f.filelist), f.filelist )

                for infor_obj in f.infolist():
                    if '__MACOSX' in infor_obj.filename:
                        continue
                    try:
                        infor_obj.filename = infor_obj.filename.encode('cp437')
                        try:
                            # infor_obj.filename = infor_obj.filename.decode('gbk')
                            infor_obj.filename = infor_obj.filename.decode('gbk').encode('utf-8').decode('utf-8')
                            # print('gbk_cracked ' * 9)
                        except Exception as e:
                            infor_obj.filename = infor_obj.filename.decode('utf-8', errors='ignore')
                            # print('brute_force_cracked ' * 10 )
                    except Exception as e:
                        pass
                    infor_obj.filename = infor_obj.filename.replace('\n', '').replace('\t', '')
                    f.extract(infor_obj, path=uz_path)

        except zipfile.BadZipfile:
            pass
        except zlib.error:
            print("unzip error : " + src + '\n', str(zlib.error))
            self.error_record("unzip error : " + src + '\n' +  str(zlib.error) + '\n')

    def un_rar(self, src, dst):
        try:
            rar = unrar.rarfile.RarFile(src)
            # print('unrar_error1 ' * 9)
            uz_path = self.zip_to_path(dst)
            # print('unrar_error2 ' * 9)
            rar.extractall(uz_path)
            # print('unrar_error3 ' * 9)
        except unrar.rarfile.BadRarFile:
            self.error_record('文件受损:' + src)
            # pass
        except Exception as e:
            # print("unrar error : " + src + '\n', str(e))
            self.error_record("unrar error : " + src + '\n' + str(e) + '\n')

    def un_tar(self, src, dst):
        try:
            tar = tarfile.open(src)
            uz_path = self.zip_to_path(dst)
            tar.extractall(path = uz_path)
        except tarfile.ReadError:
            pass
        except Exception as e:
            print("untar error : " + src + '\n', str(e))
            self.error_record("untar error : " + src + '\n' + str(e) + '\n')


class UnZip(BaseTool):
    """ UnZip files """
    def __init__(self, path1, path2):
        super().__init__(self)
        self.path = path1
        self.output = path2

    def recursive_unzip(self, repath):
        """recursive unzip file
        """
        for (root, dirs, files) in os.walk(repath):
            for filename in files:
                src = os.path.join(root,filename)
                # print('check_out_src ' * 8 + '\n', src)
                if self.iszip(src) == ".zip":
                    print("[+] child unzip: "+ src)
                    print("[+] repath_repath: "+ str(repath))
                    self.un_zip(src, src)
                    try:
                        os.remove(src)
                    except Exception as e:
                        self.error_record("remove error : " + src + '\n' + str(e) + '\n')
                    self.recursive_unzip(self.zip_to_path(src))
                    sleep(0.1)
                if self.iszip(src) == ".rar":
                    from unrar import rarfile
                    print("[+] child unrar : "+src)
                    self.un_rar(src,src)
                    try:
                        os.remove(src)
                    except Exception as e:
                        self.error_record("remove error : " + src + '\n' + str(e) + '\n')
                    self.recursive_unzip(self.zip_to_path(src))
                    sleep(0.1)
                if self.iszip(src) in (".tar.gz",".tar.bz2",".tar.bz",".tar.tgz",".tar",".tgz"):
                    print("[+] child untar : "+src)
                    self.un_tar(src,src)
                    try:
                        os.remove(src)
                    except Exception as e:
                        self.error_record("remove error : " + src + '\n' + str(e) + '\n')
                    self.recursive_unzip(self.zip_to_path(src))
                    sleep(0.1)

    def main_unzip(self):
        for (root, dirs, files) in os.walk(self.path):
            for filename in files:
                # dest_file = os.path.join(self.output, filename)
                # if not os.path.exists(dest_file):
                #     os.makedirs(dest_file)
                src = os.path.join(root, filename)
                dst = os.path.join(self.output, filename)
                if self.iszip(src) == ".zip":
                    print("[+] main unzip : " + src)
                    self.un_zip(src, dst)
                elif self.iszip(src) == ".rar":
                    from unrar import rarfile
                    print("[+] main unrar : "+src)
                    self.un_rar(src,dst)
                elif self.iszip(src) in (".tar.gz",".tar.bz2",".tar.bz",".tar.tgz",".tar",".tgz"):
                    print("[+] main untar : "+src)
                    self.un_tar(src,dst)
                else:
                    try:
                        print('shutil.copyfile(src,dst) ' * 8 + '\n' + 'src ' * 9 + '\n' + src + '\n' + 'dst ' * 9 + '\n' + dst)
                        # shutil.copyfile(src, dst)
                        shutil.copy(src, dst)
                    except OSError as e:
                        print('shutil.copyfile(src,dst) ' * 8 + '\n' + str(e) + '\n' + 'src ' * 9 + '\n' + src + '\n' + 'dst ' * 9 + dst)
                        print('self.output ' * 9 + '\n', self.output)
                        self.error_record('shutil.copyfile(src,dst) ' * 8 + '\n' + str(e))
        self.recursive_unzip(self.output)


def main():
    # filepath = Path("C:\\ChinaMobile\\cosmic_raw\\2020Q3")
    filepath = Path("C:\\ChinaMobile\\test")
    # output_path = Path("C:\\ChinaMobile\\cosmic_files\\专家评审材料_2020Q3_auto")
    output_path = Path("C:\\ChinaMobile\\test_output")
    z = UnZip(filepath, output_path)
    z.main_unzip()


if __name__ == '__main__':
    main()
