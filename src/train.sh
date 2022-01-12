source /data/envs/ai/bin/activate
cd /data/iot/notebook/src

python -u train.py h2o
python -u train.py svm
