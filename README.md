## 地学杯-地球科学大数据挑战赛

### 基于深度学习技术的智能矿物识别应用研究


* 环境安装

  python推荐版本： python2.7
  
  运行以下命令安装依赖
  ```
  pip install -r requirements.txt
  ```
* 数据集生成
  
  将`data.py`中的root改为mineral_data文件夹的地址
  
  运行以下命令生成训练和验证数据
  ```
  python data.py
  ```
* 训练
  使用默认参数训练，直接运行命令
  ```
  python train.py
  ```
* loss 和验证集准确度可视化

  ```
  python draw.py
  ```
 
  
* 测试

  在test.py中修改模型和mineral_data文件夹路径，然后运行以下命令测试

```
  python test.py
  ```

