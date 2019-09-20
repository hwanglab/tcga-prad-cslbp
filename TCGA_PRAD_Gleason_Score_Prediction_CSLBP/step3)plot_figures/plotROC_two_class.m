load(strcat('./two_class_evaluation/','label_two_class.mat'));
label=response;

load(strcat('./two_class_evaluation/','proposed_gaussian50.mat'));
p_cubic_ss=SSC;
load(strcat('./two_class_evaluation/','proposed_linear50.mat'));
p_gaussian_ss=SSC;



load(strcat('./two_class_evaluation/','fa_cubic.mat'));
fa_cubic_ss=SSC;
load('./two_class_evaluation/fa_gaussian.mat');
fa_gaussian_ss=SSC;

load(strcat('./two_class_evaluation/','hhg_cubic.mat'));
hhg_cubic_ss=SSC;
load(strcat('./two_class_evaluation/','hhg_gaussian.mat'));
hhg_gaussian_ss=SSC;

load(strcat('./two_class_evaluation/','tl_2class.mat'))
tl_ss=SSC;

load(strcat('./two_class_evaluation/','dl_2class.mat'));
dl_ss=SSC;

load(strcat('./two_class_evaluation/','hhg312_cubic.mat'));
hhg312_cubic=SSC;
load(strcat('./two_class_evaluation/','hhg312_gaussian.mat'));
hhg312_gaussian=SSC;

% 
% % ROC for g6
[x6_fa_c,y6_fa_c,t6_fa_c,auc6_fa_c]=perfcurve(label,fa_cubic_ss(:,1),'7');
[x6_fa_g,y6_fa_g,t6_fa_g,auc6_fa_g]=perfcurve(label,fa_gaussian_ss(:,1),'7');
% 
[x6_hhg_c,y6_hhg_c,t6_hhg_c,auc6_hhg_c]=perfcurve(label,hhg_cubic_ss(:,1),'7');
[x6_hhg_g,y6_hhg_g,t6_hhg_g,auc6_hhg_g]=perfcurve(label,hhg_gaussian_ss(:,1),'7');
% 
[x6_c,y6_c,t6_c,auc6_c]=perfcurve(label,p_cubic_ss(:,1),'7');
[x6_g,y6_g,t6_g,auc6_g]=perfcurve(label,p_gaussian_ss(:,1),'7');

[x_tl,y_tl,t_tl,auc_tl]=perfcurve(label,tl_ss(:,1),'7');
[x_dl,y_dl,t_dl,auc_dl]=perfcurve(label,dl_ss(:,1),'7');

[x_hhg312_c,y_hhg312_c,t_hhg312_c,auc_hhg312_c]=perfcurve(label,hhg312_cubic(:,1),'7');
[x_hhg312_g,y_hhg312_g,t_hhg312_g,auc_hhg312_g]=perfcurve(label,hhg312_gaussian(:,1),'7');


figure,
%subplot(1,3,1);
%hold on,h1=line_fewer_markers(x6_fa_c,y6_fa_c,7,'cp--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h2=line_fewer_markers(x6_fa_g,y6_fa_g,7,'p','Color',[0.8,0.5,0],'Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 

hold on, h3=line_fewer_markers(x_hhg312_g,y_hhg312_g,7,'+','Spacing', 'curve','markersize',8,'LineWidth',1.5);
%hold on,h3=line_fewer_markers(x6_hhg_c,y6_hhg_c,7,'k*--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h4=line_fewer_markers(x6_hhg_g,y6_hhg_g,7,'cd','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
hold on, h5=line_fewer_markers(x_tl,y_tl,7,'mx','Spacing', 'curve','markersize',8,'LineWidth',1.5);

hold on, h6=line_fewer_markers(x_dl,y_dl,7,'r*','Spacing', 'curve','markersize',8,'LineWidth',1.5);
%hold on,h6=line_fewer_markers(x6_c,y6_c,7,'bs--','Spacing', 'curve','markersize',8,'LineWidth',1.5);
hold on,h7=line_fewer_markers(x6_g,y6_g,7,'ks','Spacing', 'curve','markersize',8,'LineWidth',1.5);

x=0:0.05:1;
y=0:0.05:1;
hold on,h9=plot(x,y,'k:');
legend([h2,h3,h4,h5,h6,h7],{'FA (AUC=0.666)','HHG (AUC=0.726)','HHG+PCA (AUC=0.783)','VGG16-TL (AUC=0.821)','VGG16-DL (AUC=0.704)','Proposed (AUC=0.845)'});
xlabel('False positive rate'); 
ylabel('True positive rate');
ylim([0 1.03])
title('low risk and intermediate vs high risks');
grid on;

% % ROC for g7
% [x7_fa_c,y7_fa_c,t7_fa_c,auc7_fa_c]=perfcurve(label,fa_cubic_ss(:,2),'7');
% [x7_fa_g,y7_fa_g,t7_fa_g,auc7_fa_g]=perfcurve(label,fa_gaussian_ss(:,2),'7');
% 
% [x7_hhg_c,y7_hhg_c,t7_hhg_c,auc7_hhg_c]=perfcurve(label,hhg_cubic_ss(:,2),'7');
% [x7_hhg_g,y7_hhg_g,t7_hhg_g,auc7_hhg_g]=perfcurve(label,hhg_gaussian_ss(:,2),'7');
% 
% [x7_c,y7_c,t7_c,auc7_c]=perfcurve(label,p_cubic_ss(:,2),'7');
% [x7_g,y7_g,t7_g,auc7_g]=perfcurve(label,p_gaussian_ss(:,2),'7');
% 
% 
% figure,
% hold on,h1=line_fewer_markers(x7_fa_c,y7_fa_c,7,'m','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% hold on,h2=line_fewer_markers(x7_fa_g,y7_fa_g,7,'cp','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
% hold on,h3=line_fewer_markers(x7_hhg_c,y7_hhg_c,7,'gd','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% hold on,h4=line_fewer_markers(x7_hhg_g,y7_hhg_g,7,'k*','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
% hold on,h5=line_fewer_markers(x7_c,y7_c,7,'rx','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% hold on,h6=line_fewer_markers(x7_g,y7_g,7,'bs','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
% x=0:0.05:1;
% y=0:0.05:1;
% hold on,h7=plot(x,y,'k--');
% legend([h1,h2,h3,h4,h5,h6],{'FA-Polynomial','FA-Gaussian','HHG-Polynomial','HHG-Gaussian','CSLBP-Polynomial','CSLBP-Gaussian'});
% xlabel('False positive rate'); 
% ylabel('True positive rate');
% ylim([0 1.03])
% title('intermediate risk VS low and high risks');
% grid on;
% 
% 
% 
% % ROC for g8
% [x8_fa_c,y8_fa_c,t8_fa_c,auc8_fa_c]=perfcurve(label,fa_cubic_ss(:,3),'8');
% [x8_fa_g,y8_fa_g,t8_fa_g,auc8_fa_g]=perfcurve(label,fa_gaussian_ss(:,3),'8');
% 
% [x8_hhg_c,y8_hhg_c,t8_hhg_c,auc8_hhg_c]=perfcurve(label,hhg_cubic_ss(:,3),'8');
% [x8_hhg_g,y8_hhg_g,t8_hhg_g,auc8_hhg_g]=perfcurve(label,hhg_gaussian_ss(:,3),'8');
% 
% [x8_c,y8_c,t8_c,auc8_c]=perfcurve(label,p_cubic_ss(:,3),'8');
% [x8_g,y8_g,t8_g,auc8_g]=perfcurve(label,p_gaussian_ss(:,3),'8');
% 
% 
% figure,
% hold on,h1=line_fewer_markers(x8_fa_c,y8_fa_c,7,'m','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% hold on,h2=line_fewer_markers(x8_fa_g,y8_fa_g,7,'cp','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
% hold on,h3=line_fewer_markers(x8_hhg_c,y8_hhg_c,7,'gd','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% hold on,h4=line_fewer_markers(x8_hhg_g,y8_hhg_g,7,'k*','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
% hold on,h5=line_fewer_markers(x8_c,y8_c,7,'rx','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% hold on,h6=line_fewer_markers(x8_g,y8_g,7,'bs','Spacing', 'curve','markersize',8,'LineWidth',1.5);
% 
% x=0:0.05:1;
% y=0:0.05:1;
% hold on,h7=plot(x,y,'k--');
% legend([h1,h2,h3,h4,h5,h6],{'FA-Polynomial','FA-Gaussian','HHG-Polynomial','HHG-Gaussian','CSLBP-Polynomial','CSLBP-Gaussian'});
% xlabel('False positive rate'); 
% ylabel('True positive rate');
% ylim([0 1.03])
% title('high risk VS low and intermediate risks');
% grid on;
tightfig();
% 
% auc_fa_c=(auc6_fa_c+auc7_fa_c+auc8_fa_c)/3;
% auc_fa_g=(auc6_fa_g+auc7_fa_g+auc8_fa_g)/3;
% 
% auc_hhg_c=(auc6_hhg_c+auc7_hhg_c+auc8_hhg_c)/3;
% auc_hhg_g=(auc6_hhg_g+auc7_hhg_g+auc8_hhg_g)/3;
% 
% auc_cslbp_c=(auc6_c+auc7_c+auc8_c)/3;
% auc_cslbp_g=(auc6_g+auc7_g+auc8_g)/3;