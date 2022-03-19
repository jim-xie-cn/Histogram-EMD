import time,json,os
import numpy as np
import pandas as pd
import scipy.stats
from JSample import CJSample
from common import *
import seaborn as sns
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm_notebook
from warnings import filterwarnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.stats import stats
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity

class CJDistance(object):
    
    def __init__(self,list1,list2):
        self.m_list1 = list1
        self.m_list2 = list2
           
    @staticmethod
    def get_ent(data):
        p_data = data.value_counts()           # counts occurrence of each value
        entropy = scipy.stats.entropy(p_data)  # get entropy from counts
        return entropy
    
    @staticmethod
    def GetHistogram(df1,df2,bin_count=50):
        df_tmp1 = df1.copy( deep = True )
        df_tmp2 = df2.copy( deep = True )
        df_tmp = pd.concat([df_tmp1,df_tmp2],ignore_index = False)
        df_tmp = CJSample.Preprocess(df_tmp).fillna(0)
        ret_df = pd.DataFrame()
        ret_df1 = pd.DataFrame()
        ret_df2 = pd.DataFrame()
        ret_s1 = pd.Series()
        ret_s2 = pd.Series()
        for key in df_tmp.keys():
            if key in ['label']:
                continue
            #使用平滑降噪
            if key in ['in_bytes','out_bytes','in_pkts','out_pkts','duration']:
                #sli_window = df_tmp.shape[0]//bin_count
                sli_window = int(np.sqrt(tmp.shape[0])) + 1
            else:
                sli_window = 1
            df_tmp[key] = df_tmp[key].rolling(window=sli_window).mean()
            df_tmp[key] = df_tmp[key].fillna(0)
            #重新对其赋值
            tmp = df_tmp[key].head(df_tmp1.shape[0]+sli_window-1)        
            df_tmp1[key] = tmp.tail(df_tmp1.shape[0]).values
            tmp = df_tmp[key].tail(df_tmp2.shape[0])
            df_tmp2[key] = tmp
            #统一划分bin
            hist, bin_edges = np.histogram( df_tmp[key], bins = bin_count )
            hist1, bin_edges1 = np.histogram( df_tmp1[key], bins = bin_edges )
            hist2, bin_edges2 = np.histogram( df_tmp2[key], bins = bin_edges )
            #输出结果
            ret_df[key] = pd.Series(hist)
            ret_df1[key] = pd.Series(hist1)
            ret_df1[key] =(ret_df1[key]-ret_df1[key].min())/(ret_df1[key].max()-ret_df1[key].min())
            ret_s1 = ret_s1.append(ret_df1[key], ignore_index=True)
            ret_df2[key] = pd.Series(hist2)
            ret_df2[key] = (ret_df2[key] - ret_df2[key].min()) / ( ret_df2[key].max() - ret_df2[key].min() )
            ret_s2 = ret_s2.append(ret_df2[key], ignore_index=True)
        return ret_df1.T,ret_s1,ret_df2.T,ret_s2

    #余弦距离（cosine）
    def Cosine(self):
        ret = cosine_similarity([self.m_list1, self.m_list2])
        return ret[0][1]
        return cosine(self.m_list1, self.m_list2)
    
    #皮尔森相关系数（pearson）,https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    def Pearson(self):
        return stats.pearsonr(self.m_list1, self.m_list2)[0]
    
    #欧式距离
    def Euclidean(self):
        return np.linalg.norm(np.array(self.m_list1) - np.array(self.m_list2))

    #KS检验,(P越大，两个分布越相似)
    #P比指定的显著水平（假设为5%）小，则我们完全可以拒绝假设，即两个分布不服从同一分布。
    def KSTest(self):
        ret = ks_2samp(self.m_list1, self.m_list2)
        return ret[0],ret[1]
    
    #EDM距离
    def EDM(self):
        return wasserstein_distance(self.m_list1, self.m_list2)
    
    #Manhattan
    def Manhattan(self):
        return sum(abs(a-b) for a,b in zip(self.m_list1,self.m_list2))
    
    #Minkowski
    def Minkowski(self):
        return float(self.minkowski_distance(np.array(self.m_list1),np.array(self.m_list2),3))
    
    #Jaccard
    def Jaccard(self):
        return self.jaccard_similarity(self.m_list1,self.m_list2)
    
    def minkowski_distance(self,x,y,p_value):
        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
    
    def nth_root(self,value, n_root):
        root_value = 1/float(n_root)
        return round (float(value) ** float(root_value),3)
    
    def jaccard_similarity(self,x,y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)
    
    def Entropy(self):
        return abs(CJDistance.get_ent(pd.Series(self.m_list1)) - CJDistance.get_ent(pd.Series(self.m_list2)))

    #def Quadratic(self):
    #    return get_quadratic_distance(self.m_list1,self.m_list2)

    @staticmethod
    def GetDistance(df1, df2, bin_count = 50 ):
        df_tmp1,s1,df_tmp2,s2 = CJDistance.GetHistogram(df1,df2,bin_count)
        ret = []
        for key in df_tmp1.T:
            ds1 = df_tmp1.T[key]
            ds2 = df_tmp2.T[key]
            s = CJDistance(ds1.tolist(),ds2.tolist())
            tmp = {}
            tmp["Feature"] = key
            tmp['Cosine'] = s.Cosine()
            tmp['Pearson'] = s.Pearson()
            tmp['Euclidean'] = s.Euclidean()
            tmp['EMD'] = s.EDM()
            tmp['KS'] = s.KSTest()[1]
            tmp['Manhattan'] = s.Manhattan()
            tmp['Minkowski'] = s.Minkowski()
            tmp['Jaccard'] = s.Jaccard()
            tmp['Entropy'] = s.Entropy()
            ret.append(tmp)
        return pd.DataFrame(ret)

def main():
    test_file = "/data/paper4/sample/1.csv"
    df1 = pd.read_csv(test_file,index_col=0)
    test_file = "/data/paper4/sample/baseline.csv"
    df2 = pd.read_csv(test_file,index_col=0)
    tmp = CJDistance.GetDistance(df1,df2)
    print(tmp)

if __name__ == "__main__":
    main()
