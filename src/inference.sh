source /data/envs/ai/bin/activate
cd /data/iot/notebook/src

#python -u inference.py gbm
#python -u inference.py rf
#python -u inference.py bayes 
#python -u inference.py deeplearn
#python -u inference.py glm 
#python -u inference.py xgboost 
#nohup python -u inference.py svm &
nohup python -u inference.py cnn &
