## NER_Bilstm_Crf_master

### 项目介绍

​        通过对各个省/市/县/区的统计公报进行标注，识别统计公报中的时间（2020年1月）、地区（北京市海淀区）、指标（地区生产总值）、数值（10.1）、单位（亿元）。

### 环境说明

* python==3.6.12
* tensorflow==1.12.0
* numpy==1.19.2
* ...

### 更新说明

* 2020-10-27：readme.md文件增加项目运行环境的说明；

* 2020-10-27：项目上传；

### 流程说明

#### 数据

* 数据说明见Model/Parameters.py文件；
* 数据处理见Data/Data_process/handle_data.py；
* 数据格式见Data/label_data文件夹里的文件；

#### 模型

1.  模型训练：运行Model/Train.py；
2.  模型测试：运行Model/predict.py;
3.  模型评估：运行Model/score.py;





