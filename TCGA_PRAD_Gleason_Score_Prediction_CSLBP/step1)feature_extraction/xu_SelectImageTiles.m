%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for cal. the grid for given image tile size

%  Inputs:
%  -bwTissue: the binary mask with foreground region as tissue region
%  -bwHimg: the binary mask foreground as nuclei pixels region
%  -pp: the percent of the largest intrested value (the interest value is
%  the raito of nuclei pixels over the total pixels)
%  -ppt: the percent of tissue pixels in the image patch
%  -tileSize: the image tile size
%  -agumentStep: it depends on applications may be not necessary

%  Output:
%  -top_left : each row is the top-left corner of a selected block; the
%  first column: row number; the second column: column number
%  -bottome_right: each row is the bottom-right corner of a selected block



% (c) Edited by Hongming Xu,
% Deptment of Quantitative Health Sciences,
% Cleveland Clinic, USA.  December 2017
% If you have any problem feel free to contact me.
% Please address questions or comments to: mxu@ualberta.ca

% Terms of use: You are free to copy,
% distribute, display, and use this work, under the following
% conditions. (1) You must give the original authors credit. (2) You may
% not use or redistribute this work for commercial purposes. (3) You may
% not alter, transform, or build upon this work. (4) For any reuse or
% distribution, you must make clear to others the license terms of this
% work. (5) Any of these conditions can be waived if you get permission
% from the authors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [top_left,bottom_right]=xu_SelectImageTiles(bwTissue,bwHimg,pp,ppt,tileSize,augmentStep)

if nargin<5
    disp('Error: Input argments not enough!!!');
end

if nargin==5
    augmentStep=0;
end

CC=bwconncomp(bwTissue);
stats=regionprops(CC,'BoundingBox');
bb=cat(1,stats.BoundingBox);
ss=10; %% for safty not out of image, this parimeter is not affect performance
fun=@(x)sum(sum(x.data))/(tileSize(1)*tileSize(2));
top_left=[];
bottom_right=[];

if length(augmentStep)==1
    for bi=1:size(bb,1)
        tbb=bb(bi,:);
        rs=round(tbb(2))+ss+augmentStep;
        cs=round(tbb(1))+ss+augmentStep;
        re=round(tbb(2))+round(tbb(4))-ss;
        ce=round(tbb(1))+round(tbb(3))-ss;
        B=blockproc(bwHimg(rs:re,cs:ce),tileSize,fun);        %% hematoxylin channels
        Bt1=(B>max(B(:))*pp);
        BTis=blockproc(bwTissue(rs:re,cs:ce),tileSize,fun);   %% tissue binary mask
        Bt2=(BTis>ppt);
        Bt=Bt1&Bt2;
        
        yy=[rs,rs+tileSize(2):tileSize(2):re,re];           %% row direction
        xx=[cs,cs+tileSize(1):tileSize(1):ce,ce];           %% column direction
        [ry,cx]=find(Bt);
        top_left=[top_left;yy(ry)',xx(cx)'];              %% the first column: row; the second column: column; top-left point position
        bottom_right=[bottom_right;yy(ry+1)',xx(cx+1)'];  %% the first column: row; the second column: column; bottom-right point position
        
        
        %%-- for verifying our previous implementations -- %%
        %         yy=[rs,rs+tileSize(2)-1:tileSize(2):re,re]; %% row direction
        %         xx=[cs,cs+tileSize(1)-1:tileSize(1):ce,ce]; %% column direction
        %         BB=zeros(length(yy)-1,length(xx)-1);  %% for verification
        %         for yyi=1:length(yy)-1
        %             for xxi=1:length(xx)-1
        %                 if yyi==1
        %                     ssy=yy(yyi);
        %                 else
        %                     ssy=yy(yyi)+1;
        %                 end
        %                 if xxi==1
        %                     ssx=xx(xxi);
        %                 else
        %                     ssx=xx(xxi)+1;
        %                 end
        %                 temp=bwHimg(ssy:yy(yyi+1),ssx:xx(xxi+1));
        %                 BB(yyi,xxi)=sum(temp(:))/(tileSize(1)*tileSize(2));
        %             end
        %         end
        %[X,Y]=meshgrid(xx,yy);
        %hold on,plot(X,Y,'r.','MarkerSize',15,'LineWidth',5);
    end
else  %% augment data by shifting image blocks
    step=[0,augmentStep];
    for bi=1:size(bb,1)
        tbb=bb(bi,:);
        for j=1:length(step)
            rs=round(tbb(2))+ss+step(j);
            cs=round(tbb(1))+ss+step(j);
            re=round(tbb(2))+round(tbb(4))-ss;
            ce=round(tbb(1))+round(tbb(3))-ss;
            B=blockproc(bwHimg(rs:re,cs:ce),tileSize,fun);        %% hematoxylin channels
            Bt1=(B>max(B(:))*pp);
            BTis=blockproc(bwTissue(rs:re,cs:ce),tileSize,fun);   %% tissue binary mask
            Bt2=(BTis>ppt);
            Bt=Bt1&Bt2;
            
            yy=[rs,rs+tileSize(2):tileSize(2):re,re];           %% row direction
            xx=[cs,cs+tileSize(1):tileSize(1):ce,ce];           %% column direction
            [ry,cx]=find(Bt);
            top_left=[top_left;yy(ry)',xx(cx)'];              %% the first column: row; the second column: column; top-left point position
            bottom_right=[bottom_right;yy(ry+1)',xx(cx+1)'];  %% the first column: row; the second column: column; bottom-right point position
        end
    end
end
end