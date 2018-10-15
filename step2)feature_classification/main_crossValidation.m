%% The main cross validation function
% 1) load extracted features, organize them into a table structure
% 2) 5 fold cross valiation is performed by using SVM with polynomial or
% gaussian kernel
% Author: Hongming Xu, Cleveland Clinic

function main_crossValidation

% step 1) load and organize features and corresponding labels in a table
trainfeaturePath='C:\Users\xuh3\Desktop\prostate-project\TCGA_PRAD_Gleason_Score_Prediction_CSLBP\step2)feature_classification\tcga_288_25\';   % proposed cslbp feature groups
% this folder path should be changed on your machine

trainmatfile=strcat(trainfeaturePath,'*.mat');
listing=dir(trainmatfile);

Ftrain=[];  % training feature matrix
Ltrain=[];  % training lable matrix
for i=1:length(listing)
    filename=listing(i).name;
    matfilename=strcat(trainfeaturePath,filename);
    load(matfilename);
    temp=pwsif';        %%pwsif: saved extracted textural features
    Fm=cell2mat(temp);
    
    Ftrain=[Ftrain;Fm];
    if strcmp(filename(1:2),'g6')
        L1=ones(size(Fm,1),1)*6;
        Ltrain=[Ltrain;L1];
    elseif strcmp(filename(1:2),'g7')
        L2=ones(size(Fm,1),1)*7;
        Ltrain=[Ltrain;L2];
    else
        L3=ones(size(Fm,1),1)*8;
        Ltrain=[Ltrain;L3];
    end
end

FT=table(Ftrain,Ltrain);
FT.Properties.VariableNames={'features','classes'};

% step 2) cross classification evaluation
temp=zeros(1,50);           %% save 50 time cross validation accuracies
CC=zeros(3,3);              %% save confusion matrix
SSC=zeros(size(FT,1),3);    %% save classifier output probabilities which are used for generating ROC curves
for i=1:50
   %[validationAccuracy,C,scores] = trainClassifier_Cubic288_tcbb(FT);            %% SVM classifer using cubic kernel
   [validationAccuracy,C,scores] = trainClassifier_MidGaussian288_tcbb(FT);       %% SVM classifier using Gaussian kernel
   temp(i)=validationAccuracy;
   CC=CC+C;
   SSC=SSC+scores;
end
mean(temp)
CC=CC./50;
SSC=SSC./50;