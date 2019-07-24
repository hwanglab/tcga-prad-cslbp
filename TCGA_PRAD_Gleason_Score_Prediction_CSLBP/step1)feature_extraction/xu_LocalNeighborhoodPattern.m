
%% only support divide neighborhood into 8 parts
% -- input siz: the size of neighborhood
% -- output HF: 8 filters pattern
function HF=xu_LocalNeighborhoodPattern(siz)
H1=fspecial('disk',1);
H1(H1>0)=1;
H1=padarray(H1,[siz-1,siz-1]);

H4 = fspecial('disk',siz);
H4(H4>0)=1;
H4=H4-H1; %% remove the center region
HF=zeros([size(H4),8]);
ss=floor(size(H4,1)/2);
index=1;
for rr=1:2
    for cc=1:2
        temp=H4((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1);
        if rr==cc
            if rr==1
                temp(ss+1,:)=0;
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=triu(temp,1);
                index=index+1;
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=tril(temp);
                index=index+1;
            else
                temp(1,:)=0;
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=triu(temp);
                index=index+1;
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=tril(temp,-1);
                index=index+1;
            end
            
        else
            if rr==1
                temp(:,1)=0;
                temp=fliplr(temp);
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=fliplr(triu(temp));
                index=index+1;
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=fliplr(tril(temp,-1));
                index=index+1;
            else
                temp(:,ss+1)=0;
                temp=fliplr(temp);
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=fliplr(triu(temp,1));
                index=index+1;
                HF((rr-1)*ss+1:rr*ss+1,(cc-1)*ss+1:cc*ss+1,index)=fliplr(tril(temp));
                index=index+1;
            end
            
        end
    end
end