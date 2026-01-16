from JModelH2o import CJModelH2o
from JModelCNN import CJModelCNN
import json
import sys
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

def predict_one(kind,model,test_id):
    test_file = '/data/paper4/test/%s.csv'%test_id
    res = model.Predict(test_file)
    y_true = res[kind]['y_true']
    y_pred = res[kind]['y_pred']
    tmp = {}
    tmp['acccuracy'] = accuracy_score(y_true,y_pred)
    tmp['precision'] = precision_score(y_true,y_pred)
    tmp['recall'] = recall_score(y_true,y_pred)
    tmp['f1_score'] = f1_score(y_true,y_pred)
    return tmp

def predict():
    #model = CJModelCNN()
    model = CJModelH2o()
    model_b = CJModelH2o()

    kind = sys.argv[1]
    model_id = sys.argv[2]
    model.Load(model_id)
    model_b.Load("baseline")

    result = []
    if len(sys.argv) == 4:
        test_id = sys.argv[3]
        ret = predict_one(kind,model,test_id)
        tmp = {}
        tmp['train'] = model_id
        tmp['test'] = test_id
        ret_b = predict_one(kind,model_b,test_id)
        for key in ret_b:
            tmp['base-%s'%key] = ret_b[key]
            tmp['best-%s'%key] = ret[key]
        result.append(tmp)
    else:
        for i in range(9):
            test_id = "%d"%i
            ret = predict_one(kind,model,test_id)
            tmp = {}
            tmp['train'] = model_id
            tmp['test'] = test_id
            ret_b = predict_one(kind,model_b,test_id)
            for key in ret_b:
                tmp['base-%s'%key] = ret_b[key]
                tmp['best-%s'%key] = ret[key]
            result.append(tmp)
    df_result = pd.DataFrame(result)
    print(df_result.T)

def main():
    predict()

if __name__ == "__main__":
    main()
