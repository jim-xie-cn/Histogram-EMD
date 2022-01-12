source /data/env/ai/bin/activate

cd /data/iot/notebook/src

jupyter-notebook --notebook-dir=./ --ip='*' --port=80 --allow-root
