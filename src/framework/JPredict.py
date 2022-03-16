import numpy as np
import pandas as pd
import json
from common import g_predict_path,g_test_path,g_model_path,g_sample_path,get_sub_folder,get_sub_files
from JSample import CJSample
from JModelH2o import CJModelH2o
from JModelCNN import CJModelCNN

def get_models():
    ret = {}
    sample_list = get_sub_folder(g_model_path)
    for sample_name in sample_list:
        model_folder = "%s%s"%(g_model_path,sample_name)
        model_list = get_sub_folder( model_folder )
        ret[sample_name] = {}
        for model_name in model_list:
            model_files = get_sub_files("%s/%s"%(model_folder,model_name))
            for model in model_files:
                tmp = "%s/%s/%s"%(model_folder,model_name,model)
                ret[sample_name][model_name] = tmp
    return ret

def get_test():
    test_samples = get_sub_files(g_test_path)
    ret = []
    for test_name in test_samples:
        tmp = "%s%s"%(g_test_path,test_name )
        ret.append(tmp)
    return ret

def predict():
    model_info = get_models()
    test_info = get_test()
    result = []
    for sample_name in model_info:
        #if sample_name !="8":
        #    continue
        for model_name in model_info[sample_name]:
            model_file = model_info[sample_name][model_name]
            print(model_name)
            if model_name == 'cnn':
                #continue
                model = CJModelCNN()
            else:
                #continue
                model = CJModelH2o()
            model.Load(sample_name)
            for test_file in test_info:
                print(sample_name,model_name,test_file)
                predict = model.Predict(test_file)[model_name]
                y_true = predict['y_true']
                y_pred = predict['y_pred']
                tmp = {}
                tmp['train'] = sample_name
                tmp['test'] = test_file.split("/")[-1].split(".")[0]
                tmp['model'] = model_name
                tmp['y_true'] = y_true
                tmp['y_pred'] = y_pred
                result.append(tmp)
    df = pd.DataFrame(result)
    df.to_csv("%s/result.csv"%g_predict_path)
    print("predict finished")

def main():
    predict()

if __name__ == "__main__":
    main()
