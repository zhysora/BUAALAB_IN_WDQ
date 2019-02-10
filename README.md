Name Disambiguation in AMiner
============
This is implementation of our KDD'18 paper:

Yutao Zhang, Fanjin Zhang, Peiran Yao, and Jie Tang. [Name Disambiguation in AMiner: Clustering, Maintenance, and Human in the Loop](http://keg.cs.tsinghua.edu.cn/jietang/publications/kdd18_yutao-AMiner-Name-Disambiguation.pdf). In Proceedings of the Twenty-Forth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'18).

## Requirements
- Linux
- python 3
- install requirements via ```
pip install -r requirements.txt``` 

Note: Running this project will consume upwards of 10GB hard disk space. The overall pipeline will take several hours. You are recommended to run this project on a Linux server.

## How to run
```bash
cd $project_path
export PYTHONPATH="$project_path:$PYTHONPATH"

# training part
python scripts/preprocessing.py
python global_/gen_train_data.py
python global_/global_model.py
python cluster_size/count.py

# prediction part
python scripts/predict.py

```
