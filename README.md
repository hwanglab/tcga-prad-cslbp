# CSLBP Texture Descriptor

For technical descriptions of our method, please refer to the following paper: 

Hongming Xu, Sunho Park, Tae Hyun Hwang, "Computerized Classification of Prostate Cancer Gleason Scores from Whole Slide Images", TCBB 2019. (https://ieeexplore.ieee.org/document/8836110)

In the following, the steps to run our programs are provided. 

## Program Info.
The folder "TCGA_PRAD_Gleason_Score_Prediction_CSLBP" includes Matlab programs of our method, which has four subfolders:

- step1)feature_extraction
- step2)feature_classification
- step3)plot_figures
- step4)comparison

The folder "tcga_prad_tcbb_dl" mainly includes Python programs for us to test baseline deep learning models for comparison in the paper

## Usage Info.
Once your environment has met the requirments (see requirements), you could run our method by following steps:

- (1) go to the folder "step1)feature_extrction", then run the function "main_feasExtraction.m"

- - to run the "main_feasExtrcation.m", make sure that YOU should change the imageTrainPath (line 39) and imageDebugPath (line 43) to the local address of your computer
