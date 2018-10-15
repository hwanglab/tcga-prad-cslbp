function [FF3]=xu_FeatsCombination(PatF,method)

if strcmp(method,'kmeans')==1
    % for visualization  pca processing
    rpp=0.1;
    [coeff,score,latent] = pca(PatF);
    PatFP=PatF*coeff(:,1:5);
    [C,m]=covmatrix(PatFP);
    
    D=DistToCenter('mahalanobis',PatFP,m,C);
    [val,ind] = sort(D,'descend');
    nn=ceil(length(D)*rpp);
    
    %         temp=PatFP(ind(1:nn),:);
    %         figure,scatter3(PatFP(:,1),PatFP(:,2),PatFP(:,3));
    %         hold on,scatter3(m(1),m(2),m(3),'rp');
    %         hold on,scatter3(temp(:,1),temp(:,2),temp(:,3),'g');
    
    
    ind22=true(size(PatF,1),1);
    ind22(ind(1:nn))=0;
    PF22=PatF(ind22,:);
    
    PatsFm=bsxfun(@minus,PF22,mean(PF22));      % feature standardization
    PatsFs=bsxfun(@rdivide,PatsFm,std(PatsFm)); % feature standardization
    [L,Cen]=kmeansPlus(PatsFs',2);
    
    L3=ones(size(PatF,1),1)*3;
    L3(ind22)=L;
    
    FF3=bsxfun(@times,Cen',std(PatsFm));        % recover back
    FF3=bsxfun(@plus,FF3,mean(PF22));
   
elseif strcmp(method,'hist')==1                 % statistical measures used in this study
    Fmean=mean(PatF);
    Fstd=std(PatF);
    Fskew=skewness(PatF);
    Fkurt=kurtosis(PatF);
    FF3=[Fmean,Fstd,Fskew,Fkurt];
else
    disp('error input argments!!!');
end