%LBP returns the local binary pattern image or LBP histogram of an image.
%  J = LBP(I,R,N,MAPPING,MODE) returns either a local binary pattern
%  coded image or the local binary pattern histogram of an intensity
%  image I. The LBP codes are computed using N sampling points on a
%  circle of radius R and using mapping table defined by MAPPING.
%  See the getmapping function for different mappings and use 0 for
%  no mapping. Possible values for MODE are
%       'h' or 'hist'  to get a histogram of LBP codes
%       'nh'           to get a normalized histogram
%  Otherwise an LBP code image is returned.
%
%  J = LBP(I) returns the original (basic) LBP histogram of image I
%
%  J = LBP(I,SP,MAPPING,MODE) computes the LBP codes using n sampling
%  points defined in (n * 2) matrix SP. The sampling points should be
%  defined around the origin (coordinates (0,0)).
%
%  Examples
%  --------
%       I=imread('rice.png');
%       mapping=getmapping(8,'u2');
%       H1=LBP(I,1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood
%                                  %using uniform patterns
%       subplot(2,1,1),stem(H1);
%
%       H2=LBP(I);
%       subplot(2,1,2),stem(H2);
%
%       SP=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];
%       I2=LBP(I,SP,0,'i'); %LBP code image using sampling points in SP
%                           %and no mapping. Now H2 is equal to histogram
%                           %of I2.

function [LNP_S,LNP_M] = xu_OSLBP(varargin) % image,radius,neighbors,mapping,mode)
% Version 1.0.0
% Authors: Hongming Xu, Cleveland Clinic


% Check number of input arguments.
narginchk(2,5);

image=varargin{1};
d_image=double(image);

siz=varargin{2};
HF=xu_LocalNeighborhoodPattern(siz);

if nargin==2
    mapping=0;
    mode='h';
elseif nargin==3
    mapping=varargin{3};
    mode='h';
elseif nargin==4
    mapping=varargin{3};
    mode=varargin{4};
else
    mapping=varargin{3};
    mode=varargin{4};
    stat=varargin{5};
end

% Determine the dimensions of the input image.
[ysize,xsize] = size(image);


bsizey=size(HF,1);
bsizex=size(HF,2);

% Coordinates of origin (0,0) in the block
origy=floor(bsizey/2)+1;
origx=floor(bsizex/2)+1;

% Minimum allowed size for the input image depends on size of kernel
if(xsize < bsizex || ysize < bsizey)
    error('Too small input image.');
end

% Calculate dx and dy;
dx = xsize - bsizex;
dy = ysize - bsizey;

% Fill the center pixel matrix C.
C = image(origy:origy+dy,origx:origx+dx);
d_C = double(C);

neighbors=size(HF,3);
bins = 2^neighbors;

% Initialize the result matrix with zeros.
%result=zeros(dy+1,dx+1);

LNP_S=zeros(dy+1,dx+1);
LNP_M=zeros(dy+1,dx+1);
%LNP_C=zeros(dy+1,dx+1);

%Compute the LNP code image
for i = 1:neighbors
    if strcmp(stat,'mids')
        nn=floor(sum(sum(HF(:,:,i)))/2)+1;
    elseif strcmp(stat,'mins')
        nn=1;
    else
        nn=sum(sum(HF(:,:,i)));
    end
    B=ordfilt2(d_image,nn,HF(:,:,i));
    N=B(origy:origy+dy,origx:origx+dx);
    
    %     D = N >= C;
    %
    %     % Update the result matrix.
    %     v = 2^(i-1);
    %     result = result + v*D;
    
    D{i} = N >= d_C;
    Diff{i} = abs(N-d_C);
    MeanDiff(i)=mean(mean(Diff{i}));
end

% Difference threshold for CLBP_M
DiffThreshold = mean(MeanDiff);
% compute CLBP_S and CLBP_M
for i=1:neighbors
    % Update the result matrix.
    v = 2^(i-1);
    LNP_S = LNP_S + v*D{i};
    LNP_M = LNP_M + v*(Diff{i}>=DiffThreshold);
end
% CLBP_C
%LNP_C = d_C>=mean(d_image(:));
%thresh=multithresh(image,2);
%LNP_C(d_C>thresh(1))=1;
%LNP_C(d_C>thresh(2))=2;

%Apply mapping if it is defined
if isstruct(mapping)
    bins = mapping.num;
    sizarray = size(LNP_S);
    LNP_S = LNP_S(:);
    LNP_M = LNP_M(:);
    LNP_S = mapping.table(LNP_S+1);
    LNP_M = mapping.table(LNP_M+1);
    LNP_S = reshape(LNP_S,sizarray);
    LNP_M = reshape(LNP_M,sizarray);
    % % another implementation method
    
    %     for i = 1:size(result,1)
    %         for j = 1:size(result,2)
    %             result(i,j) = mapping.table(result(i,j)+1);
    %         end
    %     end
end

if (strcmp(mode,'h') || strcmp(mode,'hist') || strcmp(mode,'nh'))
    % Return with LBP histogram if mode equals 'hist'.
    %     result=hist(result(:),0:(bins-1));
    %     if (strcmp(mode,'nh'))
    %         result=result/sum(result);
    %     end
    
    LNP_S=hist(LNP_S(:),0:(bins-1));
    LNP_M=hist(LNP_M(:),0:(bins-1));
    if (strcmp(mode,'nh'))
        LNP_S=LNP_S/sum(LNP_S);
        LNP_M=LNP_M/sum(LNP_M);
    end
else
    %     %Otherwise return a matrix of unsigned integers
    %     if ((bins-1)<=intmax('uint8'))
    %         result=uint8(result);
    %     elseif ((bins-1)<=intmax('uint16'))
    %         result=uint16(result);
    %     else
    %         result=uint32(result);
    %     end
end

end



