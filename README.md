# Hybrid Network for Upper Gastrointestinal Disease Segmentation with Illumination-Boundary Modulation and Uncertainty-Aware Post-Hoc Refinement

This repository provides the official codes of DySSNet with MSM-TTA and the UGIAD-Seg dataset. This repository is under active development. Inference code and examples coming soon!


## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/allebasycin/dyssnet_msmtta.git
cd DySSNet

# Create conda environment
conda create -n dyssnet python=3.10
conda activate dyssnet

# Run automated installation
bash scripts/install.sh
```

### Manual Install
```
# Step 1: Create environment
conda create -n dyssnet python=3.10
conda activate dyssnet

# Step 2: Install PyTorch (Swin-UMamba compatible version)
pip install torch==2.0.1 torchvision==0.15.2

# Step 3: Install Mamba dependencies
pip install causal-conv1d==1.1.1
pip install mamba-ssm

# Step 4: Install other requirements
pip install -r requirements.txt
```

## Dataset Details
The UGIAD-Seg dataset provides open access to 3313 UGI endoscopic images from two hospitals, mainly captured using WLE and partly by NBI. These images encompass three key areas: esophagus, stomach, and duodenum, each annotated with specific anatomical landmarks and disease types, and these annotations are both applied and subsequently verified by medical specialists from the two contributing hospitals. The dataset is developed ensuring patient anonymity and privacy, with all materials fully anonymized by excluding patient information from the images and renaming the files according to their anatomical landmark and disease labels, and thereby exempting it from patient consent requirements. The images consist of different resolutions that range between 268x217 and 1545x1156 with most of the black borders removed. 
The dataset can also be downloaded using the following links: <br />
Google Drive: https://drive.google.com/file/d/1TioBa5SoGJF6noxPrqi0iKQkauhiIss6/view?usp=sharing <br />

### Anatomical landmark annotation
Our anatomical annotation approach is guided by previous photodocumentation guidelines such as the British and Japanese guidelines. The images are categorised into 9 landmarks. Anatomical landmarks identified in the antegrade view within the UGIAD dataset encompass the esophagus (E), squamocolumnar junction (SJ), gastric body in antegrade view (Ba), antrum (Ant), duodenal bulb (DB) and descending part of the duodenum (DD). Conversely, the retroflex view encompasses landmarks such as the fundus (F), gastric body in retroflex view (Br) and angulus (Ang).

<p align="center">
    <img src="/assets/anatomical_annotation.png" alt="Anatomical landmark annotation of the UGIAD Dataset" width="350">
</p>

### Disease annotation
For disease annotation, the images in the dataset are classified into normal findings or 8 upper gastrointestinal (UGI) diseases including esophageal neoplasm, esophageal varices, gastroesophageal reflux disease (GERD), gastric neoplasm, gastric polyp, gastric ulcer, gastric varices, and duodenal ulcer.
<p align="center">
    <img src="/assets/UGIAD_Seg_disease.png" alt="Representative images of the disease types of UGIAD-Seg." width="1200">
</p>


The following table displays the data distribution of the UGIAD dataset.
| Anatomical landmark / Disease    | Normal | Esophageal neoplasm | Esophageal varices | GERD | Gastric neoplasm | Gastric polyp | Gastric ulcer | Gastric varices | Duodenal ulcer | Total |
|----------------------------------|--------|---------------------|--------------------|------|------------------|---------------|---------------|-----------------|----------------|-------|
| Esophagus                        | 98     | 221                 | 133                | 24   | 0                | 0             | 0             | 0               | 0              | 476   |
| Squamocolumnar junction          | 96     | 35                  | 95                 | 119  | 0                | 0             | 0             | 0               | 0              | 345   |
| Fundus                           | 97     | 0                   | 0                  | 0    | 49               | 75            | 49            | 83              | 0              | 353   |
| Gastric body (antegrade)         | 166    | 0                   | 0                  | 0    | 178              | 290           | 61            | 0               | 0              | 695   |
| Gastric body (retroflex)         | 65     | 0                   | 0                  | 0    | 112              | 48            | 11            | 0               | 0              | 236   |
| Angulus                          | 87     | 0                   | 0                  | 0    | 80               | 57            | 82            | 0               | 0              | 306   |
| Antrum                           | 95     | 0                   | 0                  | 0    | 67               | 56            | 163           | 0               | 0              | 381   |
| Duodenal bulb                    | 156    | 0                   | 0                  | 0    | 0                | 0             | 0             | 0               | 197            | 353   |
| Descending part of duodenum      | 154    | 0                   | 0                  | 0    | 0                | 0             | 0             | 0               | 14             | 168   |
| Total                            | 1014   | 256                 | 228                | 143  | 486              | 526           | 366           | 83              | 211            | 3313  |


## Pretrained weights
You can download the pretrained weights of the DySSNet checkpoint, which obtains the median IoU value for the UGIAD-Seg dataet, via the below link.
| Model  | Params (M) | FLOPs (G) | IoU (%) | Link |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DySSNet  | 23.28 | 4.23 | 89.53% | [Link](https://drive.google.com/file/d/1AM9v7idLHFB0eu2SHZOtsCjR49Oo8M8g/view?usp=sharing) |


The ImageNet-1K pretrained weight that is used to train the DySSNet is also found in the pretrained weight folder of this repository or can be downloaded via this [Link](https://drive.google.com/file/d/118RJao5Qom0DJBB--vmXz-5rtPLO9j7K/view?usp=sharing).
If you want a more updated version of the ImageNet-1K pretrained weight for the Swin-UMamba† encoder, please check the official repository of Swin-UMamba and see which one is compatible with the encoder of DySSNet.