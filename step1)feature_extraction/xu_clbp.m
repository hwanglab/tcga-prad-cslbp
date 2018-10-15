function CLBP_S_MCH=xu_clbp(CLBP_S,CLBP_M,CLBP_C,num)

level=max(max(CLBP_C))+1;
% Generate histogram of CLBP_S
CLBP_SH = hist(CLBP_S(:),0:num-1);

% Generate histogram of CLBP_M
%CLBP_MH(:) = hist(CLBP_M(:),0:mapping.num-1);

% Generate histogram of CLBP_M/C
CLBP_MC = [CLBP_M(:),CLBP_C(:)];
Hist3D = hist3(CLBP_MC,[num,level]);
CLBP_MCH = reshape(Hist3D,1,numel(Hist3D));

% Generate histogram of CLBP_S_M/C
CLBP_S_MCH = [CLBP_SH,CLBP_MCH];
