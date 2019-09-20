
rng(100);
load('patInfo.mat');

load('./dl_tcbb/two_class/fold01_dl/vgg_dl_result.mat')
pred01=preds;
load('./dl_tcbb/two_class/fold02_dl/vgg_dl_result.mat');
pred02=preds;
load('./dl_tcbb/two_class/fold03_dl/vgg_dl_result.mat');
pred03=preds;

pred_all=zeros(size(patientInfo,1),2);

tt=cell2mat(pred01');
pred_all(~(sum(tt,2)==-2),:)=tt(~(sum(tt,2)==-2),:);
 
tt=cell2mat(pred02');
pred_all(~(sum(tt,2)==-2),:)=tt(~(sum(tt,2)==-2),:);

tt=cell2mat(pred03');
pred_all(~(sum(tt,2)==-2),:)=tt(~(sum(tt,2)==-2),:);

temp=pred_all(1:269,1);
pred_all(1:269,1)=pred_all(1:269,2);
pred_all(1:269,2)=temp;

cg7=0;
cg8_9=0;
num=0;
SSC=[];
for i=1:size(pred_all)
    pred_temp=pred_all(i,:);
    
    if sum(pred_temp)>0
        SSC=[SSC;pred_temp]; %% for ploting ROC curve
        num=num+1;
        [m,ind]=max(pred_temp);
        
        if i<270 && ind==1
            cg7=cg7+1;
        elseif i>269 && ind==2
            cg8_9=cg8_9+1;
        else
            disp('wrong prediction!!!');
        end
    end
    
end

acc=(cg7+cg8_9)/num;






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

