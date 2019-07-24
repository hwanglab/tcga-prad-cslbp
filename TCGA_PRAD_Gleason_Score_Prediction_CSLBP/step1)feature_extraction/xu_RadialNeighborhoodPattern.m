
%% only support divide neighborhood into 8 circles
% -- input siz: siz is a vector defining circle radius
% -- output HF: 8 filters pattern
function HF=xu_RadialNeighborhoodPattern(siz)

HF=cell(1,length(siz));
for i=1:length(siz)
    temp_siz=siz(i);
    H4 = fspecial('disk',temp_siz);
    H4(H4>0)=1;
    
    if i==1
        %H3=fspecial('disk',temp_siz-1);
        %H3(H3>0)=1;
        H3=1;
        psize=1;
    else
        H3=fspecial('disk',siz(i-1));
        H3(H3>0)=1;
        psize=siz(i)-siz(i-1);
    end
    H3=padarray(H3,[psize,psize]);
    HF{1,i}=H4-H3;
    
end