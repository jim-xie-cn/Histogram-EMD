import pandas as pd
import os,json
from sklearn.preprocessing import OneHotEncoder

g_sample_path = "/data/paper4/sample/"
g_model_path = "/data/paper4/model/"
g_test_path = "/data/paper4/test/"
g_predict_path = "/data/paper4/pred/"
g_baseline_path = "/data/paper4/baseline/"
#read all files
def get_all_files(filepath,file_list=[]):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            get_all_files(fi_d,file_list)
        else:
            file_list.append(fi_d)

def get_sub_folder(filepath):
    ret = []
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            ret.append(fi)
    return ret

def get_sub_files(filepath):
    ret = []
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if not os.path.isdir(fi_d):
            ret.append(fi)
    return ret

def main():
    file_list = []
    get_all_files("/data/paper4/model",file_list)
    print(file_list)
    folder_list = get_sub_folder("/data/paper4/model")
    print(folder_list)
    file_list = get_sub_files("/data/paper4/sample")
    print(file_list)

if __name__ == "__main__":
    main()
