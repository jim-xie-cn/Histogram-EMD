# -*- coding: utf-8 -*-
"""
    This script is used to train/inference svm models:
    Edit by Jim Xie (xiewenwei@sina.com)  2021/11/28
"""
import time,json,os
import numpy as np
import pandas as pd
from common import g_sample_path,g_model_path,get_sub_folder,get_sub_files
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from JSample import CJSample

class CJModelSVM(object):
    
    def __init__(self):
        n_estimators = 100
        self.m_clf = BaggingClassifier(SVC(probability=True,verbose=False),\
                                       max_samples=1.0 / n_estimators, \
                                       max_features=1.0, \
                                       n_estimators=n_estimators, \
                                       verbose=True,\
                                       n_jobs = 22)
    def Train(self,sample_name):
        sample_file = "%s/%s.csv"%(g_sample_path,str(sample_name))
        df = pd.read_csv(sample_file,index_col=0)
        print(sample_file)
        df = CJSample.Preprocess(df)
        start = time.time()
        X = df.copy( deep = True )
        Y = X['label']
        del X['label']
        print(Y.value_counts())
        self.m_clf.fit(X, Y)
        end = time.time()
        print("Finished Bagging SVC", end - start, self.m_clf.score(X,Y))
    
    def Save(self,model_file):
        joblib.dump(self.m_clf, model_file)
    
    def Load(self,sample_name):
        model_file = "%s%s/svm/sklearn_model_python_svm.pkl"%(g_model_path,sample_name)
        self.m_clf = joblib.load(model_file)
        print("SVM",model_file)

    def predict(self,x):
        return self.m_clf.predict(x)

    def Predict(self , sample_file ):
        df_tmp = pd.read_csv(sample_file,index_col=0)
        df_all = CJSample.Preprocess(df_tmp)
        y_true = df_all['label']
        del df_all['label']

        y_pred = self.predict(df_all)
        ret = {"svm":{}}
        ret['svm']['y_pred'] = y_pred.tolist()
        ret['svm']['y_true'] = y_true
        return ret

def train_svm(sample_name):
    csv_file = "%s/%s.csv"%(g_sample_path , sample_name)
    svm_root = "%s%s/svm/"%(g_model_path,sample_name)
    model_file = "%ssklearn_model_python_svm.pkl"%(svm_root)
    os.system("mkdir -p %s"%(svm_root))
    print("svm train begin" , sample_name)
    svm = CJModelSVM()
    svm.Train(sample_name)
    svm.Save( model_file )
    print("svm train end" , sample_name)

def main():
    sample_list = get_sub_files(g_sample_path)
    for sample_name in sample_list:
        sample_name = sample_name.split(".")[0]
        train_svm(sample_name)

if __name__ == "__main__":
    main()
