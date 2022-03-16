'''
更新FDSF距离矩阵
python JTool.py distance
调试某个模型 
python JTool.py model model_type train_set test_set
eg:
python JTool.py model rf 0 1
python JTool.py model rf 0
'''
import time,json,os
import numpy as np
import pandas as pd
import scipy.stats
from JSample import CJSample
from common import *
from JDistance import CJDistance
from tqdm import tqdm
from JModelH2o import CJModelH2o
from JModelCNN import CJModelCNN
import json
import sys
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

def predict_one(model_type,model,test_id):
    test_file = '/data/paper4/test/%s.csv'%test_id
    res = model.Predict(test_file)
    y_true = res[model_type]['y_true']
    y_pred = res[model_type]['y_pred']
    tmp = {}
    tmp['acccuracy'] = accuracy_score(y_true,y_pred)
    tmp['precision'] = precision_score(y_true,y_pred)
    tmp['recall'] = recall_score(y_true,y_pred)
    tmp['f1_score'] = f1_score(y_true,y_pred)
    return tmp

class CJTool:

    def __init__(self):
        pass

    @staticmethod
    def GetDistance(df_d):
        mask =((df_d['Feature']=='tcp_flags')|(df_d['Feature']=='src_port')|(df_d['Feature']=='dst_port')|(df_d['Feature']=='proto')|(df_d['Feature']=='l7_proto'))
        d = df_d[mask]
        d = df_d
        return json.loads(d.mean().to_json())

    @staticmethod
    def CalcDistance():
        print("Calculating distance ...")
        train_files = get_sub_files(g_sample_path)
        test_files = get_sub_files(g_test_path)
        all_similarity = []
        bar = tqdm(total=len(train_files)*len(test_files))
        for train in train_files:
            for test in test_files:
                train_file = "%s%s"%(g_sample_path,train)
                test_file = "%s%s"%(g_test_path,test)
                df_train = pd.read_csv(train_file,index_col=0)
                df_test = pd.read_csv(test_file,index_col=0)
                df_tmp = CJDistance.GetDistance(df_train,df_test,100)
                tmp = CJAnalyse.GetDistance(df_tmp)
                tmp['train'] = train.split(".")[0]
                if len(tmp['train']) == 1:
                    tmp['train'] = "0%s"%tmp['train']
                tmp['test'] = test.split(".")[0]
                if len(tmp['test']) == 1:
                    tmp['test'] = "0%s"%tmp['test']
                all_similarity.append(tmp)
                bar.update( 1 )
        df = pd.DataFrame(all_similarity)
        return df

    @staticmethod
    def Predict(model_type,train_set,test_set = None):
        if model_type == 'cnn':
            model = CJModelCNN()
            model_base = CJModelCNN()
        else:
            model = CJModelH2o()
            model_base = CJModelH2o()
        model.Load(train_set)
        model_base.Load("baseline")
        result = []
        if test_set:
            ret = predict_one(model_type,model,test_set)
            ret_base = predict_one(model_type,model_base,test_set)
            tmp = {'train':train_set,'test':test_set}
            for key in ret_base:
                tmp['base-%s'%key] = ret_base[key]
                tmp['best-%s'%key] = ret[key]
            result.append(tmp)

        else:
            for i in range(9):
                test_id = "%d"%i
                ret = predict_one(model_type,model,test_id)
                ret_base = predict_one(model_type,model_base,test_id)
                tmp = {'train':train_set,'test':test_set}
                for key in ret_base:
                    tmp['base-%s'%key] = ret_base[key]
                    tmp['best-%s'%key] = ret[key]
                    result.append(tmp)
        df_result = pd.DataFrame(result)
        print(df_result.T)

def main():
    action = sys.argv[1]
    if action == 'distance':
        df = CTool.CalcDistance()
        df.to_csv("%s/distance.csv"%g_predict_path)
        print("Finish calculating distance ...")
    elif action == 'model':
        model_type = sys.argv[2]
        train_set = sys.argv[3]
        if len(sys.argv)==5:
            test_set = sys.argv[4]
        else:
            test_set = None
        CJTool.Predict(model_type,train_set,test_set)

if __name__ == "__main__":
    main()
