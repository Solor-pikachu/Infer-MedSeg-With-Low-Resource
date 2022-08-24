Prepare for dataset
In our code, you need to prepare for tree dataset for our training dataset
1.1 a dataset that consists of 50 labeled data stores in Task777_CoarseFinetune
1.2 a dataset that consists of 50 labeled data and unlabeled data(After Crop in the abdomen) stored in Task888_FLARESSL
1.3 a dataset that consists of 50 labeled data stores in Task666_FineFinetune(After Cropping in the abdomen)

=For Training
==1.install nnUNet-master-ori. Using the first dataset for training to get the first model, and then using this model infers on 2000 unlabeled data to get the segmentation.

```python
cd nnUNet-master-ori
pip install -e.
nnUNet_plan_and_preprocess -t 777 -pl2d None
nnUNet_train 3d_lowres nnUNetTrainerV2 777 0
```

For unlabeled data
```python
nnUNet_predict -i ./data/Unlabel/images -o data/Task888/labelsTr
cp -r ./data/Unlabel/images ./data/Task888/imagesTr
```
For labeled data
```python
cp -r ./data/Label/labels ./data/Task888/labelsTr
cp -r ./data/Label/images ./data/Task888/imagesTr
```


2.install nnUNet-master-data-ssl. Perform data preprocessing on 50 label data and 2000 Unlabel data, and then train the semi-supervised model by the second dataset. Train config file(including space, Network Architecture, etc.) is in experiment_planning/experiment_planner_baseline_3DUNet, and the algorithm about semi-supervised and data split plan in training/network_training/nnUNetTrainerV2 file. 
```python
cd nnUNet-master-data-ssl
pip install -e .
```
According to abdomen segmentation(from step 1 model) crop in 2000 unlabeled data, and store CT images(After crop) in data/Task888_FLARESSL/imagesTr, 2000 Unlabeled data is inferred by step 1 model.
```python
python crop_img1.py
nnUNet_plan_and_preprocess -t 888 -pl2d None
nnUNet_train 3d_lowres nnUNetTrainerV2 888 0
```

3.install nnUNet-master-finetune-coarseseg, in this step train process, The model weight initialization for the training process comes from Task888 model which is trained by the semi-supervised algorithm.
```python
cd nnUNet-master-finetune-coarseseg
pip install -e .
nnUNet_train 3d_lowres nnUNetTrainerV2 777 0 -pretrained_weights nnUNet_trained_models/nnUNet/3d_fullres/Task888_FLARESSL/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model
```
4. install nnUNet-master-finetune-fineseg, in this step train process, The model weight initialization for the training process comes from Task666 model which is trained by the semi-supervised algorithm.
```python
cd nnUNet-master-finetune-fineseg
pip install -e .
python crop_img2.py
nnUNet_plan_and_preprocess -t 666 -pl2d None
nnUNet_train 3d_lowres nnUNetTrainerV2 666 0 -pretrained_weights nnUNet_trained_models/nnUNet/3d_fullres/Task777_CoarseFinetune/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model
```
####Important####
For Inference
How to infer in our new engine for your new model
1. We develop a new accelerated architecture about coarse to fine segmentation which designs for nnUNet inference.
1.1 Reimplemented CT images resample and Argmax algorithm to speed up on gpu and 
1.2 To avoid exceeding the video memory, we implemented the CT images segmentation Resample And Argmax algorithm on the z-axis.
1.3 We accelerate CT images in the largest connected components detection by cc3d and fastremap
```python
cd Infer
python predict.py -i input -o output
```
Our coarse and fine model Link is at:
链接: https://pan.baidu.com/s/1qJjSktn8dKhKCmzC-I5flw?pwd=lfaj 提取码: lfaj 
--来自百度网盘超级会员v2的分享
