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

  - To run the "main_feasExtrcation.m", make sure that YOU should change the imageTrainPath (line 39) and imageDebugPath (line 43) to the local address of your computer
  
  - The extracted image features are finnally saved into the .mat files, please feel free to check and revise in lines: 161-162, for your convenience
  
  Overall, by running the main_feasExtraction.m, we descrbe each WSI by a 1152 dimensional feature vector, i.e.,
  
  ![Test Image 1](https://github.com/hwanglab/tcga-prad-cslbp/blob/master/misc_figs/step1.jpg.jpeg)
  
- (2) go to the folder "step2)feature_classification", you could run our classifcation evaluations

  - In the folder tcga_288_25, it includes the .mat files with features computed by running the function "main_feasExtraction.m" on our dataset. So if you want to see our results, you could directly run the function "main_crossValiation.mat"
  
