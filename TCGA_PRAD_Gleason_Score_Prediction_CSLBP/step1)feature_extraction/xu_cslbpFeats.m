function Flbp=xu_cslbpFeats(Rimg,mapping,mapping2)

% thresh=multithresh(Rimg,2);
% LBP_C=zeros(size(Rimg));
% LBP_C(Rimg>thresh(1))=1;
% LBP_C(Rimg>thresh(2))=2;
LBP_C = Rimg>=mean(double(Rimg(:)));   %% use mean of whole image

%% a - CLBP features CLBP_S_M/C 54 features (16,2) neighborhood
rr1=2;
nei=16;
[CLBP_S,CLBP_M] = clbp(Rimg,rr1,nei,mapping,'x');
CLBP_C=LBP_C(rr1+1:size(Rimg,1)-rr1,rr1+1:size(Rimg,2)-rr1);
clbp1=xu_clbp(CLBP_S,CLBP_M,CLBP_C,mapping.num);


%% b - O-SLBP features
rr2=4;
[LNP_S1,LNP_M1]=xu_OSLBP(Rimg,rr2,mapping2,'x','mins');   %30
[LNP_S2,LNP_M2]=xu_OSLBP(Rimg,rr2,mapping2,'x','mids');   %30
[LNP_S3,LNP_M3]=xu_OSLBP(Rimg,rr2,mapping2,'x','maxs');   %30

LNP_C=LBP_C(rr2+1:size(Rimg,1)-rr2,rr2+1:size(Rimg,2)-rr2);
clnp1=xu_clbp(LNP_S1,LNP_M1,LNP_C,mapping2.num);
clnp2=xu_clbp(LNP_S2,LNP_M2,LNP_C,mapping2.num);
clnp3=xu_clbp(LNP_S3,LNP_M3,LNP_C,mapping2.num);


%% c - R-SRBP features
[RBP_S1,RBP_M1]=xu_RSLBP(Rimg,1:rr2,0,'x','mins'); % 48
[RBP_S2,RBP_M2]=xu_RSLBP(Rimg,1:rr2,0,'x','mids'); % 48
[RBP_S3,RBP_M3]=xu_RSLBP(Rimg,1:rr2,0,'x','maxs'); % 48

crbp1=xu_clbp(RBP_S1,RBP_M1,LNP_C,2^rr2);
crbp2=xu_clbp(RBP_S2,RBP_M2,LNP_C,2^rr2);
crbp3=xu_clbp(RBP_S3,RBP_M3,LNP_C,2^rr2);

Flbp=[clbp1,clnp1,clnp2,clnp3,crbp1,crbp2,crbp3];
