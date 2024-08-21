## 1. YOLOv8 

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
- run ..\AACVProject\YOLOv8\train.py
- See results in ..\AACVProject\YOLOv8\run\train

### 1.2.2 How to train the model on Kaggle
- Go to https://www.kaggle.com/code/kaiweiss/aacv-project/edit
- Click on the 'Save Version' button and let it run
- See results in https://www.kaggle.com/code/kaiweiss/aacv-project/output

### 1.3 How to validate the model
- Make sure the path to your downloaded dataset in config.yaml is correct (see 1.2.1)
- run ..\AACVProject\YOLOv8\validate.py
- See results in ..\AACVProject\YOLOv8\run\val

### 1.4 How to test the model
- Make sure the path to your downloaded dataset in config.yaml is correct (see 1.2.1)
- Hint: It's possible to run this script for videos and images
- run ..\AACVProject\YOLOv8\predict.py
- See results in ..\AACVProject\YOLOv8\run\predict

### 1.5 Results

Validation results for YOLOv8 (shrunk dataset!):

'metrics/precision(B)': 0.5814474585650324  
'metrics/recall(B)': 0.361319506335979  
'metrics/mAP50(B)': 0.38595999973569306  
'metrics/mAP50-95(B)': 0.24080287991266427  
'fitness': 0.25531859189496714

Example Test result (shrunk dataset!):

![First image test result](Results/YOLOv8/frontFar_BLR-2018-04-16_15-34-27_frontFar_0002580.jpg)
![First video result](Results/YOLOv8/video1.gif)

## 2. YOLOv8 + Weighted BCE Loss

Here we present the code for the approach described in Section **4.2** of our paper.

We conducted the following three experiments using different weights for the minority, moderate, and majority classes, along with varying the IoU thresholds for NMS:

* Strict class weights and higher IoU threshold (0.7)
* Lenient class weights and higher IoU threshold (0.7)
* Lenient class weights and lower IoU threshold (0.5)

![image](https://github.com/user-attachments/assets/9b7fd51e-0e10-41c2-b4fd-4f829f825a15)

Check the directory [Weighted BCE Loss](https://github.com/kai-weiss/AACVProject/blob/master/Weighted%20BCE%20Loss) for the code.

* Please use ultralytics (version=8.2.28) library
  ```python
  pip install ultralytics==8.2.28  

* Go to site packages in your IDE, and replace the ultralytics\utils\loss.py file with [Weighted BCE Loss/loss.py](https://github.com/kai-weiss/AACVProject/blob/master/Weighted%20BCE%20Loss/loss.py) file. Currently, the class-wise weights are set to lenient. Please check the code snippet (line 165-171 in the .py file) and use the weights you require.
  ```python
  # strict class weights
  # class_weights = torch.tensor([0.50, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 2.00, 2.00, 2.00], dtype=torch.float16).to(device)
  # lenient class weights
  class_weights = torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 2.00, 2.00, 2.00], dtype=torch.float16).to(device)
  self.bce = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction="none")

* Use the [.yaml file](https://github.com/kai-weiss/AACVProject/blob/master/Weighted%20BCE%20Loss/data/idd_data.yaml) provided. You can change the path as per your dataset path.

* After the above changes, run [Weighted BCE Loss/weightedbce_main.py](https://github.com/kai-weiss/AACVProject/blob/master/Weighted%20BCE%20Loss/weightedbce_main.py). Currently the iou threshold is set to 0.5. We experimented with the values 0.7 and 0.5 for the iou threshold.
```python
results = model.train(data="/data/idd_data.yaml", epochs=200, iou=0.5)
```

(TODO - add the confusion matrices)

The results of this approach have been described in Section **5.2** of our paper.

Out of the three experiments, lenient weights with lowert iou threshold (0.5) performed the best.

## 3. YOLOv8 + Focal Loss

Here we present the code for the approach described in Section **4.3** of our paper.

Check the directory [Focal Loss](https://github.com/kai-weiss/AACVProject/tree/master/Focal%20Loss) for the code.

* Please use ultralytics (version=8.2.28) library
  ```python
  pip install ultralytics==8.2.28

* Go to site packages in your IDE, and replace the ultralytics\utils\loss.py file with [Focal Loss/loss.py](https://github.com/kai-weiss/AACVProject/blob/master/Focal%20Loss/loss.py) file. 

* Use the [.yaml file](https://github.com/kai-weiss/AACVProject/blob/master/Focal%20Loss/data/idd_data.yaml) provided. You can change the path as per your dataset path.

* After the above changes, run [Focal Loss/focalloss_main.py](https://github.com/kai-weiss/AACVProject/blob/master/Focal%20Loss/focalloss_main.py).

(TODO - add the confusion matrices)

The results of this approach have been described in Section **5.3** of our paper.

## 4. YOLOv8 + Hierarchical Classification

First, run [predict_with_extraction_main.py](YOLOv8%2FHierarchical_classification%2Fpredict_with_extraction_main.py) to receive the original predictions.
They will be safed as "predictions.json". For now, the script is configured to run on the test datset (check [config.yaml](YOLOv8%2Fconfig.yaml)). If wanted, you can plot the result images.

Afterwards, run [hierarchical_classification_main.py](YOLOv8%2FHierarchical_classification%2Fhierarchical_classification_main.py) to get the new calculated predictions, following the hierarchical classification. 
If wanted, results will be safed as "new_predictions.json". Optionally, you can also plot the new predicions on the images.

### 4.1 View any live run that you started on Kaggle:
https://wandb.ai/kaiweiss0/projects

## 5. Ensemble Learning

Here we present the code for the approach described in Section **4.5.1** of our paper.

Check the directory [Ensemble Learning](https://github.com/kai-weiss/AACVProject/tree/master/Ensemble%20Learning) for the code. 

* The weights for the larger general model and specialized helper model are available as .pt files in [Ensemble Learning/ensemble_models](https://github.com/kai-weiss/AACVProject/tree/master/Ensemble%20Learning/ensemble_models). Use these .pt files to initialize the models in [Ensemble Learning/custom_val.py](https://github.com/kai-weiss/AACVProject/blob/master/Ensemble%20Learning/custom_val.py) file. Please note, this .py file assumes that model1 is the bigger model, and model2 is the smaller model (hence, use .pt weights accordingly). Simply run this .py file to get the evaluation metrics for the ensemble. 

* No changes need to be made to the files [Ensemble Learning/ensemble_learning.py](https://github.com/kai-weiss/AACVProject/blob/master/Ensemble%20Learning/ensemble_learning.py) and [Ensemble Learning/custom_validator.py](https://github.com/kai-weiss/AACVProject/blob/master/Ensemble%20Learning/custom_validator.py) The former .py file ensembles the prediction results from the two models of the ensemble. The latter .py file makes changes to YOLOv8's BaseValidator to generate the evaluation results for the ensemble. 

* Please use ultralytics (version=8.2.28) library as it is. 

(TODO - add the confusion matrices)

The results of this approach have been described in Section **5.5.1** of our paper.

## RESULTS

### Results: all classes (average of all classes)

![image](https://github.com/user-attachments/assets/82f601f9-aafa-4128-848c-a546449d89fc)

### Results: minority classes 

![image](https://github.com/user-attachments/assets/ad4d669c-9251-4834-9971-0bd9b260765d)
