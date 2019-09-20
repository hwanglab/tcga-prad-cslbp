function s2_cross_folds_prad

load('patInfo.mat');

labels=-1*ones(size(patientInfo,1),1);
for i=1:size(patientInfo,1)
    cc=patientInfo{i,2};
    labels(i)=cc;
end

rng(100);
cvIndices=crossvalind('Kfold',labels,3);

image_source='H:\projects\tcga_prad\tcga_prad_tcbb_dl\patches\';

%train_data_destination={'H:\projects\tcga_prad\tcga_prad_tcbb_dl\fold3\train\g6\',...
%                        'H:\projects\tcga_prad\tcga_prad_tcbb_dl\fold3\train\g7\',...
%                        'H:\projects\tcga_prad\tcga_prad_tcbb_dl\fold3\train\g8_9\'};

%-------3 class evaluation -----------
%validation_data_destination={'H:\projects\tcga_prad\tcga_prad_tcbb_dl\fold3\valid\g6\',...
%                             'H:\projects\tcga_prad\tcga_prad_tcbb_dl\fold3\valid\g7\',...
%                             'H:\projects\tcga_prad\tcga_prad_tcbb_dl\fold3\valid\g8_9\'};

%------2 class evaluation -----------
train_data_destination={'H:\projects\tcga_prad\tcga_prad_tcbb_dl\two_class\fold3\train\g7\'};

validation_data_destination={'H:\projects\tcga_prad\tcga_prad_tcbb_dl\two_class\fold3\valid\g7\'};

for i=3:3
    valid_id=(cvIndices==i);
    train_id=~valid_id;
    
    for k=1:size(patientInfo,1)
        patientID=patientInfo{k,1};
        label=patientInfo{k,2};
        
        if strcmp(patientID(end),'1') %%------2 class evaluation -----------
            temp=dir(strcat(image_source,patientID,'*'));
            if length(temp)>100
                disp(patientID);
            end
            
            if valid_id(k)==1
                switch label
                    case 6
                        xu_copy_images(image_source,validation_data_destination{1},temp);
                    case 7
                        %xu_copy_images(image_source,validation_data_destination{2},temp);
                    case 8
                        %xu_copy_images(image_source,validation_data_destination{3},temp);
                    otherwise
                        disp('impossible');
                end
            else
                switch label
                    case 6
                        xu_copy_images(image_source,train_data_destination{1},temp);
                    case 7
                        %xu_copy_images(image_source,train_data_destination{2},temp);
                    case 8
                        %xu_copy_images(image_source,train_data_destination{3},temp);
                    otherwise
                        disp('impossible');
                end
            end
        end
        
    end
end