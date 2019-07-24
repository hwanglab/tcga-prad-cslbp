
p_g_3class=[74.13,76.27,77.12,75.82,76.41];

p_c_3class=[73.35,75.17,76.48,76.62,76.25];

%p_c_2class=[69.59,72.29,75.12,73.85,73.95];
p_c_3_noaug=[67.21,68.60,70.28,70.21,70.88];

%p_g_2class=[73.54,74.48,76.36,74.28,74.42];
p_g_3_noaug=[69.88,71.61,72.24,72.14,72.44];

x=[30,40,50,60,70];

figure,
plot(x,p_c_3class,'r-s',x,p_g_3class,'g-*',x,p_c_3_noaug,'b-o',x,p_g_3_noaug,'k-x','LineWidth',2,'MarkerSize',10)
xticks([30,40,50,60,70])
xlabel('Number of PCA components');
ylabel('Classification accuracy');
legend('Proposed-Polynomial (with augmentation)','Proposed-Gaussian (with augmentation)','Proposed-Polynomial (no augmentation)','Proposed-Gaussian (no augmentation)');
grid on;
tightfig();