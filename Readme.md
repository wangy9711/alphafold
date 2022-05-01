## command
```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 /home/ictsun/alphafold/run_multimer.py --input_fasta=/home/ictsun/data/test/diabody_test.fasta --output_dir=/home/ictsun/data/test 
```

## 必选参数说明
- ```CUDA_VISIBLE_DEVICES=0```用于指定显卡ID，需要指定
- ```XLA_PYTHON_CLIENT_PREALLOCATE=false```用于指定显卡显存的申请策略，一般不用更改
- ```--input_fasta=xx```指定输入文件
- ```--output_dir=xxx```指定输出目录


## 可选参数说明
- ```--recycle=3``` 指定推理时使用的recycle次数，默认为3
- ```--use_gpu_relax=True``` 指定openMM使用的平台，默认为CUDA，可选为CPU。
- ```--single_data_dir=xxx```指定AF2 数据库
- ```--multimer_data_dir=xxx```指定Multimer新增的几个数据库
- ```--param_dir=xxx``` 指定模型参数目录