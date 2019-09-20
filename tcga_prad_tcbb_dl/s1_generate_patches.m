%% main function for extracting CSLBP features from TCGA prostate cancer whole slide images
% to run the program:
% you must install: matlab image processing toolbox
% you must install: openslide-matlab from https://github.com/fordanic/openslide-matlab
% you must download TCGA prostate cancer whole slide images into your computer

close all;clc;

% this is the path for matlab-openslide, which should be changed based on
% your computer's path
addpath(genpath('E:\matlab_repository\toolboxes\openslide-matlab-master\'));
addpath(genpath('H:\projects\tcga_prad\TCGA_PRAD_Gleason_Score_Prediction_CSLBP\step1)feature_extraction\'));

%%--main parameter settings --%%
magCoarse = 2.5;     %% resolution for coarse processing tile selection
magFine = 5.0;       %% resoluton for fine processing image tiles
tileSize=[128 128];  %% tile size since we process at coarse resolution 2.5
ss=10;               %% for safty not out of image boundary, this parimeter does not affect performance
debug=0;             %% debug or not
thrWhite=220;        %% white foreground threshold
pp=0.25;             %% choose blocks with nuclei pixels above than 0.25(best)% of top value ???? make sure patches have nuclei
ppt=0.9;             %% choose blocks with tissue pixels above than 90% of block size ???? make sure pathes mainly have tissues
Toverstain=0.35*255; %% pixels below this threshold considered as overstaining empirally determined value but general for different images
augmentStep=[0,30,60,90];                   %% augmenting steps
%%-- end parameter setting -- %%


% image path in our computer, it should be changed if you test it on your dataset
imageTrainPath={'H:\projects\tcga_prad\data\g6Images\','H:\projects\tcga_prad\data\g7Images\',...
    'H:\projects\tcga_prad\data\g8Images\','H:\projects\tcga_prad\data\g9Images\'};

% deug path in our computer
imgpath='H:\projects\tcga_prad\tcga_prad_tcbb_dl\patches\';

patpath='H:\projects\tcga_prad\tcga_prad_tcbb_dl\';
patientInfo=[];
pid=1;
tic
for gc=1:length(imageTrainPath)
    imagePath=imageTrainPath{gc};
    imgs=dir(fullfile(imagePath,'*.svs'));
    
    for k=1:numel(imgs)
        file1=fullfile(imagePath,imgs(k).name);
        
        fprintf('fileName=%s\n%d\n',file1,k);
        
        slidePtr=openslide_open(file1);
        % Get whole-slide image properties
        [mppX, mppY, width, height, numberOfLevels, ...
            downsampleFactors, objectivePower] = openslide_get_slide_properties(slidePtr);
        
        %%1) read the 2.5x image or less resolution for coarse processing
        mag=objectivePower./round(downsampleFactors);
        xPos=1;yPos=1;
        if any(mag==magCoarse)   %2.5x magnification
            levelToUse = find(mag == magCoarse,1);
            wid3=round(width/downsampleFactors(levelToUse));
            hei3=round(height/downsampleFactors(levelToUse));
            ARGB = openslide_read_region(slidePtr,xPos,yPos,wid3,hei3,levelToUse-1);
            RGB=ARGB(:,:,2:4);
        else   %% there is no 2.5x magnification in original image
            magToUseBelow=max(mag(mag<magCoarse));
            magToUseAbove=min(mag(mag>magCoarse)); %% for reading image patches in high resolution
            if isempty(magToUseBelow)              %% there is not lower resolution than 2.5x
                levelToUse=find(mag==magToUseAbove,1);
                wid3=round(width/downsampleFactors(levelToUse));
                hei3=round(height/downsampleFactors(levelToUse));
                ARGB = openslide_read_region(slidePtr,xPos,yPos,wid3,hei3,levelToUse-1);
                RGB=ARGB(:,:,2:4);
                RGB=imresize(RGB,magCoarse/magToUseAbove); %% reduce to 2.5 for preprocessing
            else
                levelToUse=find(mag == magToUseBelow,1);
                wid3=round(width/downsampleFactors(levelToUse));
                hei3=round(height/downsampleFactors(levelToUse));
                ARGB = openslide_read_region(slidePtr,xPos,yPos,wid3,hei3,levelToUse-1);
                RGB=ARGB(:,:,2:4);
                RGB=imresize(RGB,magCoarse/magToUseBelow);
            end
        end
        
        
        %%2) image preprocessing module & select interested regions
        gimg=rgb2gray(RGB);
        bwTissue=(gimg<=thrWhite);
        bwTissue=bwareaopen(bwTissue,tileSize(1)*tileSize(2)*6);        % step 1: obtain tissue pixels
        %--fill small holes in the image--%
        bwNoholes=imfill(bwTissue,'holes');
        holes=bwNoholes&~bwTissue;
        bigholes=bwareaopen(holes,round(tileSize(1)*tileSize(2)/3));    % holes greater than half of image patch not filled
        smallholes=holes&~bigholes;
        bwTissue=bwTissue|smallholes;
        %-- end filling small holes -- %
        
        RGB=imresize(RGB,0.5);                                   % for memory issue
        [~,Himg,~]=normalizeStaining(RGB);                       % step 2: color normalization
        gHimg=rgb2gray(Himg);
        gHimg=imresize(gHimg,size(bwTissue));                    % for memory issue
        thresh=multithresh(gHimg,2);
        bwHimgN=(gHimg<=thresh(1));                              % step 3: obtain nuclei regions
        bwHimgO=(gHimg>Toverstain);                              % over staining regions
        bwHimg=bwHimgN&bwHimgO;
        RGB=imresize(RGB,2);                                     % recover the size
        
        
        %%3) feature extraction module
        if gc==1 % in this study: for the Gleason score g6 images they are augmented by shifting bounding box
            for aug=1:length(augmentStep)
                [top_left,bottom_right]=xu_SelectImageTiles(bwTissue,bwHimg,pp,ppt,tileSize,augmentStep(aug));
                
                if debug==1  % for debuging: see selected image tiles highlighted on wsi
                    xu_debugShownTiles(RGB,bwTissue,top_left,tileSize);
                end
                
                ind=length(augmentStep)*(k-1)+aug;
                if any(mag==magFine)
                    levelforRead=find(mag==magFine,1);
                    
                    prad_generate_patches(top_left,bottom_right,slidePtr,levelforRead,magFine,magCoarse,strcat(imgs(k).name(1:23),'_',num2str(aug)),imgpath);
                    
                else %% for reading image patches in high resolution
                    magToUseAbove=min(mag(mag>magFine));
                    levelforRead=find(mag==magToUseAbove);
                    prad_generate_patches(top_left,bottom_right,slidePtr,levelforRead,magFine,magCoarse,strcat(imgs(k).name(1:23),'_',num2str(aug)),imgpath,magToUseAbove);
                end
                
                
                patientInfo{pid,1}=strcat(imgs(k).name(1:23),'_',num2str(aug));
                patientInfo{pid,2}=6;
                pid=pid+1;
            end
            
            
            
        else   % for g7, g8 g9 and g10 images no image augmentation is performed
            [top_left,bottom_right]=xu_SelectImageTiles(bwTissue,bwHimg,pp,ppt,tileSize);
            
            if debug==1 % for debuging: see selected image tiles highlighted on wsi
                xu_debugShownTiles(RGB,bwTissue,top_left,tileSize);
            end
            
            ind=k;
            if any(mag==magFine)
                levelforRead=find(mag==magFine,1);
                prad_generate_patches(top_left,bottom_right,slidePtr,levelforRead,magFine,magCoarse,imgs(k).name(1:23),imgpath);
                
            else %% for reading image patches in high resolution
                magToUseAbove=min(mag(mag>magFine));
                levelforRead=find(mag==magToUseAbove);
                prad_generate_patches(top_left,bottom_right,slidePtr,levelforRead,magFine,magCoarse,imgs(k).name(1:23),imgpath,magToUseAbove);
            end
            
            patientInfo{pid,1}=strcat(imgs(k).name(1:23));
            if gc==2
                patientInfo{pid,2}=7;
            else
                patientInfo{pid,2}=8;
            end
            pid=pid+1;
            
        end
    end
    
    
    
end
%save(strcat(patpath,'patInfo.mat'),'patientInfo');                    % save the output features
toc
tt=0;
