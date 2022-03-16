'''
python JPerformance.py sync
python JPerformance.py report
'''
import time,json,os,sys
import numpy as np
import pandas as pd
from JSample import CJSample
from JDistance import CJDistance
from common import *
import seaborn as sns
from tqdm.notebook import tqdm
from warnings import filterwarnings
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)
plt.rcParams['axes.unicode_minus'] = False

class CJAnalyse:

    def __init__(self):
        self.m_raw_result = pd.read_csv("%sresult.csv"%g_predict_path,index_col=0)
        self.m_raw_similarity = pd.read_csv("%sdistance.csv"%g_predict_path,dtype={'test':object,"train":object},index_col=0)
        self.m_result = pd.DataFrame()
        
    def CalcResult(self):
        df = self.m_raw_result
        all_result = []
        bar = tqdm(total=df.shape[0]-1)
        for model,train_set,data_set,yt,yp in zip(df['model'],df['train'],df['test'],df['y_true'],df['y_pred']):
            y_true = json.loads(yt)
            y_pred = json.loads(yp)
            tmp = {}
            tmp['model'] = model
            if  not train_set in ['baseline','reserve']:
                tmp['train'] = "%02d"%(int(train_set))
            else:
                tmp['train'] = train_set
            if not data_set in ['baseline','reserve']:
                tmp['test'] = "%02d"%(int(data_set))
            else:
                tmp['test'] = data_set
            tmp['acccuracy'] = accuracy_score(y_true,y_pred)
            tmp['precision'] = precision_score(y_true,y_pred)
            tmp['recall'] = recall_score(y_true,y_pred)
            tmp['f1_score'] = f1_score(y_true,y_pred)
            all_result.append(tmp)
            bar.update(1)
        self.m_result = pd.DataFrame(all_result)
        return self.m_result
    
    def SaveResult(self):
        df = self.m_result
        df.to_csv("%sanalyse.csv"%g_predict_path)
    
    def LoadResult(self):
        self.m_result = pd.read_csv("%sanalyse.csv"%g_predict_path,dtype={'test':object,"train":object},index_col=0)
        return self.m_result
    
    def FindTrainSample(self,test_name):
        df = self.m_raw_similarity.copy(deep = True )
        mask = (df['test']==test_name)&(df['train']!="baseline")
        tmp = df[mask].sort_values(by=['Cosine']).tail(3)
        tmp = tmp[mask].sort_values(by=['EMD'])
        tmp = tmp.head( 1 )
        return tmp
    
    def FindBest(self,model,test_name):
        df_tmp = self.m_result[self.m_result['model'] == model].copy(deep = True )
        mask = ( df_tmp['train'] == 'baseline' ) & ( df_tmp['test'] == test_name )
        df_baseline = df_tmp[mask]
        sample = self.FindTrainSample(test_name)
        mask1 = pd.Series()
        for train_sample in sample['train']:
            if mask1.any():
                mask1 = ( mask1 ) | ( df_tmp['train'] == train_sample )
            else:
                mask1 = ( df_tmp['train'] == train_sample )
        mask1 = (mask1) & ( df_tmp['test'] == test_name)
        df_best = df_tmp[mask1]
        return sample,df_baseline.reset_index(drop=True), df_best.reset_index(drop=True)

def set_kind(train_set):
    if train_set == 'baseline':
        return 'baseline'
    else:
        return "HDFS"

def main():
    analyse = CJAnalyse()
    analyse.LoadResult()

    df_report = pd.DataFrame()
    df_baseline = pd.DataFrame()
    df_best = pd.DataFrame()
    for model in analyse.m_result['model'].unique():
        #if model == 'cnn':
        #    continue
        for i in range(9):
            sample,baseline,best = analyse.FindBest(model,"%02d"%i)
            df_baseline = pd.concat([df_baseline,baseline],ignore_index = True)
            df_best = pd.concat([df_best,best],ignore_index = True)
            df_report = pd.concat([df_baseline,df_best],ignore_index = True)
    df_report['kind'] = df_report.apply(lambda x:set_kind(x['train']),axis=1)
    df_report.rename(columns={"acccuracy":"accuracy"},inplace=True)

    df_compare = df_report.groupby(['kind','model']).mean().reset_index()
    df_best = df_compare[df_compare['kind']=="HDFS"].sort_values(by='model',ascending=False).reset_index(drop=True)
    df_base = df_compare[df_compare['kind']!="HDFS"].sort_values(by='model',ascending=False).reset_index(drop=True)
    df_diff = pd.DataFrame()
    df_diff['model'] = df_best['model']
    for measure in ['accuracy','precision','recall','f1_score']:
        df_diff[measure] = df_best[measure] - df_base[measure]
    del df_best['kind']
    del df_base['kind']
    print(df_base,df_best,df_diff,df_diff.mean())

if __name__ == "__main__":
    action = sys.argv[1]
    if action == 'sync':
        analyse = CJAnalyse()
        analyse.CalcResult()
        analyse.SaveResult()
    elif action == 'report':
        main()
