
testing=1; % with augmentation

if testing==1
    % %% testing on 408 patients with augmentation
    load('patInfo.mat');
    
    load('../models/tf_v2/fold01/vgg_tf2_fold01.mat')
    pred01=preds;
    
    
    load('../models/tf_v2/fold02/vgg_tf2_fold02.mat');
    pred02=preds;
    
    %load('../models/fold03/vggresult_fold03.mat')
    load('../models/tf_v2/fold03/vgg_tf2_fold03.mat');
    pred03=preds;
    
    pred_all=zeros(size(patientInfo,1),3);
    
    tt=cell2mat(pred01');
    pred_all(~(sum(tt,2)==-3),:)=tt(~(sum(tt,2)==-3),:);
    
    tt=cell2mat(pred02');
    pred_all(~(sum(tt,2)==-3),:)=tt(~(sum(tt,2)==-3),:);
    
    tt=cell2mat(pred03');
    pred_all(~(sum(tt,2)==-3),:)=tt(~(sum(tt,2)==-3),:);
    
    cg6=0;
    cg7=0;
    cg8_9=0;
    num=0;
    for i=1:size(pred_all)
        pred_temp=pred_all(i,:);
        
        if sum(pred_temp)>0
            num=num+1;
            [m,ind]=max(pred_temp);
            
            if i <129 && ind==1
                cg6=cg6+1;
            elseif i>128 && i<270 && ind==2
                cg7=cg7+1;
            elseif i>269 && ind==3
                cg8_9=cg8_9+1;
            else
                disp('wrong prediction!!!');
            end
        end
        
    end
    
    acc=(cg6+cg7+cg8_9)/num;
else
    %% testing on 312 patients without augmentation
    load('patInfo.mat');
    
    %load('../models/fold01/vggresult_fold01.mat')
    load('../models/tf_v2/fold01/vgg_tf2_fold01.mat')
    pred01=preds;
    
    %load('../models/fold02/vggresult_fold02.mat')
    load('../models/tf_v2/fold02/vgg_tf2_fold02.mat');
    pred02=preds;
    
    load('../models/fold03/vggresult_fold03.mat')
    %load('../models/tf_v2/fold03/vgg_tf2_fold03.mat');
    pred03=preds;
    
    pred_all=zeros(size(patientInfo,1),3);
    
    tt=cell2mat(pred01');
    pred_all(~(sum(tt,2)==-3),:)=tt(~(sum(tt,2)==-3),:);
    
    tt=cell2mat(pred02');
    pred_all(~(sum(tt,2)==-3),:)=tt(~(sum(tt,2)==-3),:);
    
    tt=cell2mat(pred03');
    pred_all(~(sum(tt,2)==-3),:)=tt(~(sum(tt,2)==-3),:);
    
    pred_all2=[pred_all(1:4:128,:);pred_all(129:end,:)];
    
    cg6=0;
    cg7=0;
    cg8_9=0;
    num=0;
    for i=1:size(pred_all2)
        pred_temp=pred_all2(i,:);
        
        num=num+1;
        [m,ind]=max(pred_temp);
        
        if i <33 && ind==1
            cg6=cg6+1;
        elseif i>32 && i<174 && ind==2
            cg7=cg7+1;
        elseif i>173 && ind==3
            cg8_9=cg8_9+1;
        else
            disp('wrong prediction!!!');
        end
        
    end
    
    acc=(cg6+cg7+cg8_9)/num;
end



% cg6=0;
% cg7=0;
% cg8_9=0;
% num=0;
% for i=1:length(preds)
%     pred_temp=preds{i};
%     if sum(pred_temp)>0
%         num=num+1;
%         [m,ind]=max(pred_temp);
%
%         if i <129 && ind==1
%             cg6=cg6+1;
%         elseif i>128 && i<270 && ind==2
%             cg7=cg7+1;
%         elseif i>269 && ind==3
%             cg8_9=cg8_9+1;
%         else
%             disp('wrong prediction!!!');
%         end
%     end
% end
%
% acc=(cg6+cg7+cg8_9)/num;

