import numpy as np
import pandas as pd
import json,os
from common import g_predict_path,g_test_path,g_model_path,g_sample_path,get_sub_folder,get_sub_files
from JSample import CJSample
from JModelH2o import CJModelH2o,train_h2o
from JModelCNN import CJModelCNN,train_cnn
from JModelSVM import CJSVM,train_svm

def main():
    sample_list = get_sub_files(g_sample_path)
    for sample_name in sample_list:
        sample_name = sample_name.split(".")[0]
        train_h2o(sample_name)
        #train_cnn(sample_name)
        #train_svm(sample_name)

if __name__ == "__main__":
    main()
