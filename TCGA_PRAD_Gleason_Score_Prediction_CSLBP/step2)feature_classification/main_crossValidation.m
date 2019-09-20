%% The main cross validation function
% 1) load extracted features, organize them into a table structure
% 2) 5 fold cross valiation is performed by using SVM with polynomial or
% gaussian kernel
% Author: Hongming Xu, Cleveland Clinic

function main_crossValidation

%rng(100);
% step 1) load and organize features and corresponding labels in a table
trainfeaturePath='.\tcga_288_25\';   % proposed cslbp feature groups
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
    
    
    if strcmp(filename(1:2),'g6')
        %Fm2=Fm(1:4:128,:);
        L1=ones(size(Fm,1),1)*6;
        Ltrain=[Ltrain;L1];
        Ftrain=[Ftrain;Fm];
    elseif strcmp(filename(1:2),'g7')
        L2=ones(size(Fm,1),1)*7;
        Ltrain=[Ltrain;L2];
        Ftrain=[Ftrain;Fm];
    else
        L3=ones(size(Fm,1),1)*8;
        Ltrain=[Ltrain;L3];
        Ftrain=[Ftrain;Fm];
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
   %[validationAccuracy,C,scores] = trainClassifier_MidGaussian288_tcbb(FT);       %% SVM classifier using Gaussian kernel
   [validationAccuracy,C,scores] = trainClassifier_MidGaussian_tcbb_noAug(FT);
   %[validationAccuracy,C,scores] = trainClassifier_Cubic_tcbb_noAug(FT);
   temp(i)=validationAccuracy;
   CC=CC+C;
   SSC=SSC+scores;
end
mean(temp)
CC=CC./50;
SSC=SSC./50;
SSC2=[SSC(1:4:128,:);SSC(129:end,:)];