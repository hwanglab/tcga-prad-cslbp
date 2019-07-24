%% The main cross validation function
% 1) load extracted features, organize them into a table structure
% 2) 5 fold cross valiation is performed by using SVM with polynomial or
% gaussian kernel
% in this program, we only test 2 class classification: g7 vs >g7
% Author: Hongming Xu, Cleveland Clinic

function cross_testing_fa_twoClass

% step 1) load and organize features and corresponding labels in a table
trainfeaturePath='./fa8/';   % proposed cslbp feature groups
% this folder path should be changed on your machine

trainmatfile=strcat(trainfeaturePath,'*.mat');
listing=dir(trainmatfile);

Ftrain=[];  % training feature matrix
Ltrain=[];  % training lable matrix
for i=1:length(listing)
    filename=listing(i).name;
    %if ~strcmp(filename,'g6Images.mat') % two class testing excluding g6 patients which are augmented
        matfilename=strcat(trainfeaturePath,filename);
        load(matfilename);
        temp=pwsif';        %%pwsif: saved extracted textural features
        Fm=cell2mat(temp);
        
        if strcmp(filename(1:2),'g6')
            Fm2=Fm(1:4:128,:);
            L1=ones(size(Fm2,1),1)*7;
            Ltrain=[Ltrain;L1];
            Ftrain=[Ftrain;Fm2];
        elseif strcmp(filename(1:2),'g7')
            L2=ones(size(Fm,1),1)*7;
            Ltrain=[Ltrain;L2];
            Ftrain=[Ftrain;Fm];
        else
            L3=ones(size(Fm,1),1)*8;
            Ltrain=[Ltrain;L3];
            Ftrain=[Ftrain;Fm];
        end
    %end
end

FT=table(Ftrain,Ltrain);
FT.Properties.VariableNames={'features','classes'};

% step 2) cross classification evaluation
temp=zeros(1,50);           %% save 50 time cross validation accuracies
CC=zeros(2,2);              %% save confusion matrix
SSC=zeros(size(FT,1),2);    %% save classifier output probabilities which are used for generating ROC curves
for i=1:50
    [validationAccuracy,C,scores] = trainClassifier_twoclass_fa(FT);
    temp(i)=validationAccuracy;
    CC=CC+C;
    SSC=SSC+scores;
end
mean(temp)
CC=CC./50;
SSC=SSC./50;
%save('fa_linear.mat','SSC','CC');