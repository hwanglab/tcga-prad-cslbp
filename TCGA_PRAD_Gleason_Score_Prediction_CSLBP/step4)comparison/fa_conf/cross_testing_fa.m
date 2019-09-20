function cross_testing_fa

%% FT.mat saves the features and labels which are organized as below.
%load('FT.mat'); %table FT is organized with features stored in fold fa8

%rng(100);
% step 1) load and organize features and corresponding labels in a table
trainfeaturePath='.\fa8\';   % proposed cslbp feature groups
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

%% classification evaluation
temp=zeros(1,50);
CC=zeros(3,3);
SSC=zeros(size(FT,1),3);
for i=1:50
   %[~, validationAccuracy,C,scores] = trainClassifier_fa_cubic8(FT);  %% box: 1 kenrel:3
   %[~, validationAccuracy,C,scores] = trainClassifier_fa_gaussian8(FT);  %% box: 1 kenrel:3
   %[~, validationAccuracy,C,scores] = trainClassifier_fa_cubic8_noAug(FT);
   [~, validationAccuracy,C,scores] = trainClassifier_fa_gaussian8_noAug(FT);
   temp(i)=validationAccuracy;
   CC=CC+C;
   SSC=SSC+scores;
end
mean(temp)
CC=CC./50;
SSC=SSC./50;
SSC2=[SSC(1:4:128,:);SSC(129:end,:)];