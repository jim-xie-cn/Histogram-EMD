import numpy as np
import pandas as pd
from common import g_sample_path,g_model_path,get_sub_folder,get_sub_files
from JSample import CJSample
import json,os
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ORandomForestEstimator

class CJModelH2o:
    
    def __init__(self):
        h2o.init(ip="localhost",port=54321)
        self.m_models = {
            "bayes":H2ONaiveBayesEstimator(),
            "glm":H2OGeneralizedLinearEstimator(nfolds = 4),
            "rf":H2ORandomForestEstimator(nfolds = 4),
            "gbm":H2OGradientBoostingEstimator(nfolds=4),
            #"svm":H2OSupportVectorMachineEstimator(),
            "xgboost":H2OXGBoostEstimator(nfolds=4),
            "deeplearn":H2ODeepLearningEstimator(hidden=[100, 100],nfolds = 4 ,force_load_balance = False)
        }

    def get_performance(self,model_name,df_test):
        result = {}
        model = self.m_models[model_name]
        perf = model.model_performance(df_test)
        result['mcc'] = json.loads(str(perf.mcc()))
        result['f1'] = json.loads(str(perf.F1()))
        result['f05'] = json.loads(str(perf.F0point5()))
        result['f2'] = json.loads(str(perf.F2()))
        result['accuracy'] =json.loads(str(perf.accuracy()))
        result['logloss'] = json.loads(str(perf.logloss()))
        result['recall'] = json.loads(str(perf.recall()))
        result['precision'] = json.loads(str(perf.precision()))
        result['gini'] = perf.gini()
        result['auc'] = perf.auc()
        result['aucpr'] = perf.aucpr()
        result['roc'] = json.loads(json.dumps(perf.roc()))
        result['fpr'] = json.loads(json.dumps(perf.fpr()))
        result['tpr'] = json.loads(json.dumps(perf.tpr()))
        result['confusion_matrix'] = perf.confusion_matrix().to_list()
        return result

    def Train(self,sample_name,model_name = None):
        sample_file = "%s/%s.csv"%(g_sample_path,str(sample_name))
        df = pd.read_csv(sample_file,index_col=0)
        print(sample_file)
        df_all = CJSample.Preprocess(df)
        df_h2o = h2o.H2OFrame(df_all)
        df_train, df_valid = df_h2o.split_frame(ratios=[0.85], seed=1234)
        x = df_train.columns
        y = "label"
        x.remove(y)
        df_train[y] = df_train[y].asfactor()
        df_valid[y] = df_valid[y].asfactor()
        meta_info = {}
        for key in self.m_models:
            if (not model_name) or (key == model_name):
                print("begin train ",key)
                self.m_models[key].train(x=x, y=y,training_frame=df_train,validation_frame=df_valid)
                model_path = "%s/%s/%s"%(g_model_path,str(sample_name),key)
                os.system("rm -rf %s"%model_path)
                model_file = h2o.save_model(model=self.m_models[key], path=model_path, force=True)
                print("end train ",key,model_file)
                meta_info[key] = {}
                meta_info[key]['sample_name'] = str(sample_name)
                meta_info[key]['performance'] = self.get_performance(key,df_valid)
                meta_info[key]['model_file'] = model_file

        with open("%s/%s/meta.info"%(g_model_path,str(sample_name)),"w") as fp:
            data = json.dumps(meta_info,indent=4)
            fp.write(data)
        print("train finished")

    def Load(self,sample_name):
        for key in self.m_models:
            model_path = "%s/%s/%s/"%(g_model_path,str(sample_name),key)
            tmp = get_sub_files(model_path)
            tmp.sort(reverse=False)
            for item in tmp:
                model_file = "%s%s"%(model_path,item)
                print("loading ",key,model_file)
                self.m_models[key] =  h2o.load_model(model_file)
                break

    def Predict(self , sample_file ):
        df_tmp = pd.read_csv(sample_file,index_col=0)
        df_all = CJSample.Preprocess(df_tmp)
        df_h2o = h2o.H2OFrame(df_all)
        x = df_h2o.columns
        y = "label"
        x.remove(y)
        df_h2o[y] = df_h2o[y].asfactor()
        ret = {}
        for key in self.m_models:
            pred = self.m_models[key].predict(df_h2o).as_data_frame()
            y_true = df_all[y].to_list()
            y_pred = pred['predict']
            ret[key] = {}
            ret[key]['y_pred'] = y_pred.to_list()
            ret[key]['y_true'] = y_true
        return ret

def train_h2o(sample_name):
    model = CJModelH2o()
    model.Train(sample_name)

def train():
    sample_list = get_sub_files(g_sample_path)
    for sample_name in sample_list:
        sample_name = sample_name.split(".")[0]
        train_h2o(sample_name)

def predict():
    model = CJModelH2o()
    meta = model.Load("0")
    sample_file = "/data/paper4/test/0.csv"
    result = model.Predict(sample_file)
    print(json.dumps(result,indent=4))

def main():
    train_h2o("1")
    #train()
    #predict()

if __name__ == "__main__":
    main()
