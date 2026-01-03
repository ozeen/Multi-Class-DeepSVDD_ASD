Multi-Class Deep SVDD for Anomalous Sound Detection(ASD)


### 1) 训练 | Training
- 直接运行 `train.py`：  
  Run `train.py` directly:
  ```bash
  python train.py
  
如需修改训练参数，请编辑 config/config.yaml：
To change training settings, edit config/config.yaml.


### 2)数据集文件夹结构示例| Example dataset folder structure:
data/dataset/fan
data/dataset/slider

data/ 文件夹与代码文件夹放在同一根目录下：
Place the data/ folder under the same project root as the code.

若修改数据路径，请同时修改 config.yaml 中对应的路径参数：
If you change the dataset path, also update the corresponding path fields in config.yaml.

### 3)测试| test:
python test.py --weight_dir "权重路径"
python test.py --weight_dir "path/to/weights"

### 4) 集成测试 | Ensemble Testing
python ensemble_test.py --ensemble_weight_dir "集成权重文件夹路径"
python ensemble_test.py --ensemble_weight_dir "path/to/ensemble_weights_dir"
