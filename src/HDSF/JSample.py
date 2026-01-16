import numpy as np
import pandas as pd
import random
from common import g_sample_path,g_test_path,g_baseline_path

class CJSample:
    
    def __init__(self):
        self.m_df_data = pd.DataFrame()
        self.m_df_reserve = pd.DataFrame()

    def Save(self,train_file_name,test_file_name):
        self.m_df_train.to_csv(train_file_name)
        self.m_df_test.to_csv(test_file_name)

    def Load_NF_ToN_IoT(self,file_name):
        df = pd.read_csv(file_name)
        #df = df[(df['Attack'] == 'ddos') | (df['Label']==0)]
        df_common = pd.DataFrame()
        df_common['src_port'] = df['L4_SRC_PORT'].astype('int64')
        df_common['dst_port'] = df['L4_DST_PORT'].astype('int64')
        df_common['proto'] = df['PROTOCOL'].astype('int64')
        df_common['l7_proto'] = df['L7_PROTO'].astype('float')
        df_common['in_bytes'] = df['IN_BYTES'].astype('int64')
        df_common['out_bytes'] = df['OUT_BYTES'].astype('int64')
        df_common['in_pkts'] = df['IN_PKTS'].astype('int64')
        df_common['out_pkts'] = df['OUT_PKTS'].astype('int64')
        df_common['duration'] = df['FLOW_DURATION_MILLISECONDS'].astype("float")
        df_common['tcp_flags'] = df['TCP_FLAGS'].astype("int64")
        df_common['label'] = df['Label'].astype("int64")
        #df_common['src_ip'] = df['IPV4_SRC_ADDR']
        #df_common['dst_ip'] = df['L4_SRC_PORT']
        #df_common['attack'] = df['Attack'].astype("category")
        '''
        df_common['duration'] = (df_common['duration']/1000).replace(np.inf,0).astype('float')
        df_common['in_bytes'] = (df_common['in_bytes']/df_common['duration']).replace(np.inf,0).astype('float')
        df_common['out_bytes'] = (df_common['out_bytes']/df_common['duration']).replace(np.inf,0).astype('float')
        df_common['in_pkts'] = (df_common['in_pkts']/df_common['duration']).replace(np.inf,0).astype('float')
        df_common['out_pkts'] = (df_common['out_pkts']/df_common['duration']).replace(np.inf,0).astype('float')
        '''
        self.m_df_data = df_common
        #.head(9*df.shape[0]//10)
        #self.m_df_reserve = df_common.tail(df.shape[0]//10)
        print(df.shape)
    def Load_NF_ToN_IoT_V2(self,file_name):
        df = pd.read_csv(file_name)
        df = df[(df['Attack'] == 'ddos') | (df['Label']==0)]
        df_common = pd.DataFrame()
        df_common['src_port'] = df['L4_SRC_PORT'].astype('int64')
        df_common['dst_port'] = df['L4_DST_PORT'].astype('int64')
        df_common['proto'] = df['PROTOCOL'].astype('int64')
        df_common['l7_proto'] = df['L7_PROTO'].astype('float')
        df_common['in_bytes'] = df['IN_BYTES'].astype('int64')
        df_common['out_bytes'] = df['OUT_BYTES'].astype('int64')
        df_common['in_pkts'] = df['IN_PKTS'].astype('int64')
        df_common['out_pkts'] = df['OUT_PKTS'].astype('int64')
        df_common['duration'] = df['FLOW_DURATION_MILLISECONDS'].astype("float")
        df_common['tcp_flags'] = df['TCP_FLAGS'].astype("int64")
        df_common['label'] = df['Label'].astype("int64")
        #df_common['src_ip'] = df['IPV4_SRC_ADDR']
        #df_common['dst_ip'] = df['L4_SRC_PORT']
        #df_common['attack'] = df['Attack'].astype("category")
        '''
        df_common['duration'] = (df_common['duration']).replace(np.inf,0).astype('float')
        df_common['in_bytes'] = (df_common['in_bytes']/df_common['duration']).replace(np.inf,0).astype('float')
        df_common['out_bytes'] = (df_common['out_bytes']/df_common['duration']).replace(np.inf,0).astype('float')
        df_common['in_pkts'] = (df_common['in_pkts']/df_common['duration']).replace(np.inf,0).astype('float')
        df_common['out_pkts'] = (df_common['out_pkts']/df_common['duration']).replace(np.inf,0).astype('float')
        '''
        self.m_df_data = df_common.head(9*df.shape[0]//10)
        self.m_df_reserve = df_common.tail(df.shape[0]//10)

    def GetSampleRate(self):
        json_count = self.m_df_data['label'].value_counts()
        normal_count = json_count[0]
        abnormal_count = json_count[1]
        all_count = normal_count + abnormal_count
        return normal_count/all_count,abnormal_count/all_count

    def GetSample(self,frac_or_n,normal_rate,is_replace):
        raw_normal_rate,raw_abnormal_rate = self.GetSampleRate()
        df_tmp = self.m_df_data.copy(deep=True)
        df_tmp['freq'] = df_tmp['label']
        mask = (df_tmp['label']==0)
        df_tmp.loc[mask, 'freq'] = raw_abnormal_rate*(normal_rate)
        mask = (df_tmp['label']!=0)
        df_tmp.loc[mask, 'freq'] = raw_normal_rate*(1-normal_rate)
        print("get sample ",frac_or_n,normal_rate,1-normal_rate)
        if frac_or_n <= 1:
            return self.m_df_data.sample(frac=frac_or_n,weights=df_tmp['freq'].values,replace = is_replace)
        else:
            return self.m_df_data.sample(n=frac_or_n,weights=df_tmp['freq'].values, replace = is_replace)
    
    def GetTrainSet(self):
        rate_list = np.linspace(start=0.1,stop=0.9,num=9)
        ret_dataset = []
        for rate in rate_list:
            df_ds = self.GetSample(1/len(rate_list),rate,False)
            ret_dataset.append(df_ds)
        return ret_dataset

    def GetTestSet(self):
        ret_dataset = []
        rate_list = np.linspace(start=0.1,stop=0.9,num=9)
        for rate in rate_list:
            frac_or_n = random.randint( 10000*5 , 10000* 6)
            print("sampling:",frac_or_n,rate)
            df_ds = self.GetSample(frac_or_n, rate, False)
            ret_dataset.append(df_ds)
        return ret_dataset

    def GetReserveSet(self):
        return self.m_df_reserve

    @staticmethod 
    def Preprocess(df_data):
        df_ret = df_data.copy(deep = True)
        #df_ret['label'] = df_ret['label'].map(lambda x: 1 if x.lower() != 'benign' else 0)
        #df_ret['duration'] = df_ret['duration']
        #df_ret['in_bytes'] = (df_ret['in_bytes']/df_ret['duration']).replace(np.inf,0).astype('float')
        #df_ret['out_bytes'] = (df_ret['out_bytes']/df_ret['duration']).replace(np.inf,0).astype('float')
        #df_ret['in_pkts'] = (df_ret['in_pkts']/df_ret['duration']).replace(np.inf,0).astype('float')
        #df_ret['out_pkts'] = (df_ret['out_pkts']/df_ret['duration']).replace(np.inf,0).astype('float')

        if 'index' in df_ret.keys():
            del df_ret['index']
        df_ret.fillna(0)
        all_keys = df_data.keys().tolist()
        for key in all_keys:
            if key in ['label','index']:
                continue
            '''
            std = df_ret[key].std()
            if std != 0 :
                df_ret[key] = df_ret[key]/std
            if (df_ret[key].max()  - df_ret[key].min()) > 0 :
                df_ret[key] = (df_ret[key] - df_ret[key].min())/ (df_ret[key].max() - df_ret[key].min())
            else:
                df_ret[key] = df_ret[key].max()
            '''
        return df_ret

def create_train():
    root_set = CJSample()
    root_set.Load_NF_ToN_IoT_V2('/data/dataset/test/NF-ToN-IoT-v2/dataset/NF-ToN-IoT-v2.csv')
    #root_set.Load_NF_ToN_IoT('/data/dataset/test/NF-ToN-IoT/dataset/NF-ToN-IoT.csv')
    root_set.GetSampleRate()
    normal_rate,abnormal_rate = root_set.GetSampleRate()
    print("normal rate",normal_rate,abnormal_rate)

    sub_dataset = root_set.GetTrainSet()
    baseline_dataset = pd.DataFrame()
    for i in range(len(sub_dataset)):
        df = sub_dataset[i]
        print("create dataset",i)
        baseline_dataset = pd.concat([baseline_dataset,df],ignore_index=True)
        df.to_csv("%s%d.csv"%(g_sample_path,i))
    baseline_dataset = baseline_dataset.reset_index()
    if 'index' in baseline_dataset.keys():
        del baseline_dataset['index']
    baseline_dataset.to_csv("%sbaseline.csv"%g_sample_path)
    print("create train set done")

def create_test():
    root_set = CJSample()
    root_set.Load_NF_ToN_IoT('/data/dataset/test/NF-ToN-IoT/dataset/NF-ToN-IoT.csv')
    #root_set.Load_NF_ToN_IoT('/data/dataset/test/NF-ToN-IoT-v2/dataset/NF-ToN-IoT-v2.csv')
    sub_dataset = root_set.GetTestSet()
    baseline_dataset = pd.DataFrame()
    for i in range(len(sub_dataset)):
        df = sub_dataset[i]
        print("create dataset",i)
        baseline_dataset = pd.concat([baseline_dataset,df],ignore_index=True)
        df.to_csv("%s%d.csv"%(g_test_path,i) )
    baseline_dataset = baseline_dataset.reset_index()
    if 'index' in baseline_dataset.keys():
        del baseline_dataset['index']
    baseline_dataset.to_csv("%sbaseline.csv"%g_baseline_path)
    print("create test set done")

def create_reserver():
    root_set = CJSample()
    root_set.Load_NF_ToN_IoT_V2('/data/dataset/test/NF-ToN-IoT-v2/dataset/NF-ToN-IoT-v2.csv')
    df_reserver = root_set.GetReserveSet()
    if 'index' in df_reserver.keys():
        del df_reserver_dataset['index']
    df_reserver.to_csv("%sreserve.csv"%g_baseline_path)
    print("create reserve set done")

def main():
    print("create samples")
    #create_train()
    create_test()
    #create_reserver()

if __name__ == "__main__":
    main()
