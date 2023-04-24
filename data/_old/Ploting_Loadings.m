%Ploting Loadings


% Let's plot the loadings for the  for the first canonical variate (aka
% component), the only significan canonical variate in your analysis

% Model 1
figure;
subplot(1,2,1);
bar(M{1,1}.cca.XL(:,1));
subplot(1,2,2);
bar(M{1,1}.cca.YL(:,1));

% Model 2
figure;
subplot(1,2,1);
bar(M{1,2}.cca.XL(:,1));
subplot(1,2,2);
bar(M{1,2}.cca.YL(:,1));

xbars2 = (M{1,2}.cca.XL(:,1))*-1;
ybars2 = (M{1,2}.cca.YL(:,1))*-1;

figure;
subplot(1,2,1);
bar(xbars2);
subplot(1,2,2);
bar(ybars2);

% Model 3
figure;
subplot(1,2,1);
bar(M{1,3}.cca.XL(:,1));
subplot(1,2,2);
bar(M{1,3}.cca.YL(:,1));
