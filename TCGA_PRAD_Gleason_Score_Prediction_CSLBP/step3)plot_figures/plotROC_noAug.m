load('label.mat');
label0=response;
label=[label0(1:4:128);label0(129:end)];


load(strcat('./three_class_noaug/','proposed_cubic.mat'));
p_cubic_ss=SSC2;

load(strcat('./three_class_noaug/','proposed_gaussian.mat'));
p_gaussian_ss=SSC2;

load(strcat('./three_class_noaug/','fa_cubic.mat'));
fa_cubic_ss=SSC2;
load(strcat('./three_class_noaug/','fa_gaussian.mat'));
fa_gaussian_ss=SSC2;

load(strcat('./three_class_noaug/','hhg_pca_cubic.mat'));
hhg_cubic_ss=SSC2;
load(strcat('./three_class_noaug/','hhg_pca_gaussian.mat'));
hhg_gaussian_ss=SSC2;

load('tf_3class.mat');
tfpred0=pred_all;
tfpred=[tfpred0(1:4:128,:);tfpred0(129:end,:)];

load('dl_3class.mat');
dlpred0=pred_all;
dlpred=[dlpred0(1:4:128,:);dlpred0(129:end,:)];

load(strcat('./three_class_noaug/','hhg_cubic.mat'));
hhg312_cubic=SSC2;

load(strcat('./three_class_noaug/','hhg_gaussian.mat'));
hhg312_gaussian=SSC2;


% ROC for g6
[x6_fa_c,y6_fa_c,t6_fa_c,auc6_fa_c]=perfcurve(label,fa_cubic_ss(:,1),'6');
[x6_fa_g,y6_fa_g,t6_fa_g,auc6_fa_g]=perfcurve(label,fa_gaussian_ss(:,1),'6');

[x6_hhg_c,y6_hhg_c,t6_hhg_c,auc6_hhg_c]=perfcurve(label,hhg_cubic_ss(:,1),'6');
[x6_hhg_g,y6_hhg_g,t6_hhg_g,auc6_hhg_g]=perfcurve(label,hhg_gaussian_ss(:,1),'6');

[x6_c,y6_c,t6_c,auc6_c]=perfcurve(label,p_cubic_ss(:,1),'6');
[x6_g,y6_g,t6_g,auc6_g]=perfcurve(label,p_gaussian_ss(:,1),'6');

[xt,yt,tt,auct]=perfcurve(label,tfpred(:,1),'6');
[xd,yd,td,aucd]=perfcurve(label,dlpred(:,1),'6');

[x_hhg_c,y_hhg_c,t6_hhg312,auc6_c_312]=perfcurve(label,hhg312_cubic(:,1),'6');
[x_hhg_g,y_hhg_g,t6_hhg312_g,auc6_g_312]=perfcurve(label,hhg312_gaussian(:,1),'6');


figure,
%subplot(1,3,1);
%hold on,h1=line_fewer_markers(x6_fa_c,y6_fa_c,7,'gp--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h2=line_fewer_markers(x6_fa_g,y6_fa_g,7,'p','Color',[0.8,0.5,0],'Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on,h3=line_fewer_markers(x_hhg_g,y_hhg_g,7,'+','Spacing', 'curve','markersize',8,'LineWidth',1.5);

%hold on,h3=line_fewer_markers(x6_hhg_c,y6_hhg_c,7,'bd--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h4=line_fewer_markers(x6_hhg_g,y6_hhg_g,7,'cd','Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on, h5=line_fewer_markers(xt,yt,7,'mx','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on, h6=line_fewer_markers(xd,yd,7,'r*','Spacing', 'curve','markersize',8,'LineWidth',1.5);

%hold on,h7=line_fewer_markers(x6_c,y6_c,7,'ks--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h8=line_fewer_markers(x6_g,y6_g,7,'ks','Spacing', 'curve','markersize',8,'LineWidth',1.5);

x=0:0.05:1;
y=0:0.05:1;
hold on,h9=plot(x,y,'k--');
legend([h2,h3,h4,h5,h6,h8],{'FA','HHG','HHG+PCA','VGG16-TL','VGG16-DL','Proposed'});
xlabel('False positive rate'); 
ylabel('True positive rate');
ylim([0 1.03])
title('low risk vs intermediate and high risks');
grid on;
tightfig();

% ROC for g7
[x7_fa_c,y7_fa_c,t7_fa_c,auc7_fa_c]=perfcurve(label,fa_cubic_ss(:,2),'7');
[x7_fa_g,y7_fa_g,t7_fa_g,auc7_fa_g]=perfcurve(label,fa_gaussian_ss(:,2),'7');

[x7_hhg_c,y7_hhg_c,t7_hhg_c,auc7_hhg_c]=perfcurve(label,hhg_cubic_ss(:,2),'7');
[x7_hhg_g,y7_hhg_g,t7_hhg_g,auc7_hhg_g]=perfcurve(label,hhg_gaussian_ss(:,2),'7');

[x7_c,y7_c,t7_c,auc7_c]=perfcurve(label,p_cubic_ss(:,2),'7');
[x7_g,y7_g,t7_g,auc7_g]=perfcurve(label,p_gaussian_ss(:,2),'7');

[xt7,yt7,tt7,auct7]=perfcurve(label,tfpred(:,2),'7');
[xd7,yd7,td7,aucd7]=perfcurve(label,dlpred(:,2),'7');

[x7_312_c,y7_312_c,t7_hhg312,auc7_c_312]=perfcurve(label,hhg312_cubic(:,2),'7');
[x7_312_g,y7_312_g,t7_hhg312_g,auc7_g_312]=perfcurve(label,hhg312_gaussian(:,2),'7');

figure,
%hold on,h1=line_fewer_markers(x7_fa_c,y7_fa_c,7,'gp--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h2=line_fewer_markers(x7_fa_g,y7_fa_g,7,'p','Color',[0.8,0.5,0],'Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on,h3=line_fewer_markers(x7_312_g,y7_312_g,7,'+','Spacing', 'curve','markersize',8,'LineWidth',1.5);

%hold on,h3=line_fewer_markers(x7_hhg_c,y7_hhg_c,7,'bd--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h4=line_fewer_markers(x7_hhg_g,y7_hhg_g,7,'cd','Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on, h5=line_fewer_markers(xt7,yt7,7,'mx','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on, h6=line_fewer_markers(xd7,yd7,7,'r*','Spacing', 'curve','markersize',8,'LineWidth',1.5);

%hold on,h7=line_fewer_markers(x7_c,y7_c,7,'ks--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h8=line_fewer_markers(x7_g,y7_g,7,'ks','Spacing', 'curve','markersize',8,'LineWidth',1.5);

x=0:0.05:1;
y=0:0.05:1;
hold on,h9=plot(x,y,'k--');
legend([h2,h3,h4,h5,h6,h8],{'FA','HHG','HHG+PCA','VGG16-TL','VGG16-DL','Proposed'});
xlabel('False positive rate'); 
ylabel('True positive rate');
ylim([0 1.03])
title('intermediate risk vs low and high risks');
grid on;
tightfig();


% ROC for g8
[x8_fa_c,y8_fa_c,t8_fa_c,auc8_fa_c]=perfcurve(label,fa_cubic_ss(:,3),'8');
[x8_fa_g,y8_fa_g,t8_fa_g,auc8_fa_g]=perfcurve(label,fa_gaussian_ss(:,3),'8');

[x8_hhg_c,y8_hhg_c,t8_hhg_c,auc8_hhg_c]=perfcurve(label,hhg_cubic_ss(:,3),'8');
[x8_hhg_g,y8_hhg_g,t8_hhg_g,auc8_hhg_g]=perfcurve(label,hhg_gaussian_ss(:,3),'8');

[x8_c,y8_c,t8_c,auc8_c]=perfcurve(label,p_cubic_ss(:,3),'8');
[x8_g,y8_g,t8_g,auc8_g]=perfcurve(label,p_gaussian_ss(:,3),'8');

[xt8,yt8,tt8,auct8]=perfcurve(label,tfpred(:,3),'8');
[xd8,yd8,td8,aucd8]=perfcurve(label,dlpred(:,3),'8');

[x8_312_c,y8_312_c,t8_hhg312,auc8_c_312]=perfcurve(label,hhg312_cubic(:,3),'8');
[x8_312_g,y8_312_g,t8_hhg312_g,auc8_g_312]=perfcurve(label,hhg312_gaussian(:,3),'8');

figure,
%hold on,h1=line_fewer_markers(x8_fa_c,y8_fa_c,7,'gp--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h2=line_fewer_markers(x8_fa_g,y8_fa_g,7,'p','Color',[0.8,0.5,0],'Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on,h3=line_fewer_markers(x8_312_g,y8_312_g,7,'+','Spacing', 'curve','markersize',8,'LineWidth',1.5);

%hold on,h3=line_fewer_markers(x8_hhg_c,y8_hhg_c,7,'bd--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h4=line_fewer_markers(x8_hhg_g,y8_hhg_g,7,'cd','Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on, h5=line_fewer_markers(xt8,yt8,7,'mx','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on, h6=line_fewer_markers(xd8,yd8,7,'r*','Spacing', 'curve','markersize',8,'LineWidth',1.5);

%hold on,h7=line_fewer_markers(x8_c,y8_c,7,'ks--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h8=line_fewer_markers(x8_g,y8_g,7,'ks','Spacing', 'curve','markersize',8,'LineWidth',1.5);

x=0:0.05:1;
y=0:0.05:1;
hold on,h9=plot(x,y,'k--');
legend([h2,h3,h4,h5,h6,h8],{'FA','HHG','HHG+PCA','VGG16-TL','VGG16-DL','Proposed'});
xlabel('False positive rate'); 
ylabel('True positive rate');
ylim([0 1.03])
title('high risk vs low and intermediate risks');
grid on;

tightfig();

auc_fa_c=(auc6_fa_c+auc7_fa_c+auc8_fa_c)/3;
auc_fa_g=(auc6_fa_g+auc7_fa_g+auc8_fa_g)/3;

auc_hhg_c=(auc6_hhg_c+auc7_hhg_c+auc8_hhg_c)/3;
auc_hhg_g=(auc6_hhg_g+auc7_hhg_g+auc8_hhg_g)/3;

auc_cslbp_c=(auc6_c+auc7_c+auc8_c)/3;
auc_cslbp_g=(auc6_g+auc7_g+auc8_g)/3;

auc_tl=(auct+auct7+auct8)/3;
auc_dl=(aucd+aucd7+aucd8)/3;

auc_hhg312_c=(auc6_c_312+auc7_c_312+auc8_c_312)/3;
auc_hhg312_g=(auc6_g_312+auc7_g_312+auc8_g_312)/3;


% auc_cslbp_c=(auc6_c+auc7_c+auc8_c)/3;
% auc_cslbp_g=(auc6_g+auc7_g+auc8_g)/3;