
p_g_3class=[71.90,78.57,77.78,75.27,72.69];

p_c_3class=[77.84,77.24,76.95,76.11,74.49];

%p_c_2class=[73.83,74.08,73.92,75.38,75.68]; 
p_c_3_noaug=[72.11,71.32,70.51,70.53,69.06];

%p_g_2class=[55.46,73.42,75.10,75.01,75.52];
p_g_3_noaug=[65.22,73.23,73.38,70.35,69.33];
x=[3,5,7,9,11];

figure,
plot(x,p_c_3class,'r-s',x,p_g_3class,'g-*',x,p_c_3_noaug,'b-o',x,p_g_3_noaug,'k-x','LineWidth',2,'MarkerSize',10)
xticks([3,5,7,9,11])
xlabel('Kernel scales');
ylabel('Classification accuracy');
legend('Proposed-Polynomial (with augmentation)','Proposed-Gaussian (with augmentation)','Proposed-Polynomial (no augmentation)','Proposed-Gaussian (no augmentation)');
grid on;
tightfig();