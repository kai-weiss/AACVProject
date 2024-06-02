# 1. YOLOv8 

## 1.1 Installation
(If you want to use this on Kaggle, this step can be skipped)
```
git clone https://github.com/kai-weiss/AACVProject.git
cd AACVProject/YOLOv8
pip install -r requirements.txt
```

## 1.2 Training
### 1.2.1 How to train the model locally
- Download the dataset under https://kaggle.com/datasets/fcd636ded04af634ec210e3a5316c42837e2220cb794a84e31fae2808e565f8a
  (Old dataset with 15 classes: https://kaggle.com/datasets/0a201d5cb8eba5c1a719be8d390f5af4b93aa7a4057db962afbb4b08fb5183ec )
- Under ..\AACVProject\YOLOv8\config.yaml: Modify the path to the path of your downloaded dataset
- run ..\AACVProject\YOLOv8\main.py

### 1.2.2 How to train the model on Kaggle
- Go to https://www.kaggle.com/code/kaiweiss/aacv-project/edit
- Click on the 'Run All' button
- If the console asks you to enter an API key, paste this in: 1ee681c23fb80ab32f209b543ef19111269008b2

### 1.3 How to test the model
- TODO -


# 2. YOLOv8 + CBAM

# 3. YOLOv8 + CBAM + RL

# 4. Others

### 4.1 View any live run that you started on Kaggle:
https://wandb.ai/kaiweiss0/projects
