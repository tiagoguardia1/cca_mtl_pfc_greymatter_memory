% code to perform model comparisons for multivariate models with different
% complexity, similar to Tsvetanov et al 2018
% https://pubmed.ncbi.nlm.nih.gov/30049889/
clear;
%kat_import('multivista');
%rootdir      = '/imaging/camcan/sandbox/kt03/projects/collabs/karen/Tiago/cca_model_comparisons/';
%rootdir = '/mnt/data/Tiago/ModelFitKamen';
rootdir = 'C:\_ARQ\05.Neuro\03.PhD\OneDrive - Brock University\05.Thesis\02.ProjectPhD\03.1-CamCAN MTL\08.Analysis\Analysis11\CCAPROOF94\proof92-cross_perm';
path2data    = fullfile(rootdir,'data');
path2results = fullfile(rootdir,'new_memory');


%perc = participants 270, 218 and 124 were removed

load(fullfile(path2data,'assoc_val_new.mat'));
load(fullfile(path2data,'cca9a.mat'));
load(fullfile(path2data,'cca9b.mat'));
load(fullfile(path2data,'cca9c.mat'));
load(fullfile(path2data,'age_quali.mat'));

M{1}.X = cca9a_xm307;
M{1}.Y = cca9_ycog307_item_assoc_average;
M{2}.X = cca9b_xf307;
M{2}.Y = cca9_ycog307_item_assoc_average;
M{3}.X = cca9c_xmf307;
M{3}.Y = cca9_ycog307_item_assoc_average;


M{1}.X = cca9a_xm307;
M{1}.Y = cca9_ycog307_assoc_val;
M{2}.X = cca9b_xf307;
M{2}.Y = cca9_ycog307_assoc_val;
M{3}.X = cca9c_xmf307;
M{3}.Y = cca9_ycog307_assoc_val;

%% CCA BOOTSTRAP
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------%%
% Y = score(:,1);t
numboot = 5000;
% M1 = bootstrp(numboot, @myccabootstr,M{1}.X,M{1}.Y); % < -- This works
% M2 = bootstrp(numboot, @myccabootstr,M{2}.X,M{2}.Y); % < -- This works
% M3 = bootstrp(numboot, @myccabootstr,M{3}.X,M{3}.Y); % < -- This works

Mboot = bootstrp(numboot, @myccabootstr_cmp,M{1}.X,M{2}.X,M{3}.X,M{1}.Y); % < -- This works
figure;histogram([Mboot.fval1]);hold on;histogram([Mboot.fval2]);histogram([Mboot.fval3])


figure;histogram([Mboot.fval1]);
hold on;histogram([Mboot.fval2]);
histogram([Mboot.fval3]);
legend('M1','M2','M3');

% Alternative colors for the histogram

[0.8500 0.3250 0.0980]
[0.4940 0.1840 0.5560]
[0 0.4470 0.7410]


figure;histogram([Mboot.fval1]);
hold on;histogram([Mboot.fval2]);
histogram([Mboot.fval3]);
legend('M1','M2','M3');




% Compare F values (Update variables)

% -------------------------------------
% Compare Partitionining distribituions
% -------------------------------------
%cfg       =[];
%cfg.type  = 'one-sample';
%cfg.data  = [Mboot.fval1,Mboot.fval2,Mboot.fval3];
%cfg.nperm = 1000;
%cfg       = kat_stats_perm_maxT(cfg);
%tval      = cfg.tstat
%pvals     = cfg.p_orig
%pPerm     = cfg.p_perm

% Compare model 1 with model 2
cfg       =[];
cfg.type  = 'two-sample-paired-ttest';%one-sample two-sample-paired
cfg.data  = [[Mboot.fval1];[Mboot.fval2]]';
cfg.nperm = [10000];
compare_mod1_mod2 = kat_stats_perm_maxT(cfg);
%tval      = cfg.tstat
%pvals     = cfg.p_orig
%pPerm     = cfg.p_perm

% Compare model 1 with model 3
cfg       =[];
cfg.type  = 'two-sample-paired-ttest';%one-sample two-sample-paired
cfg.data  = [[Mboot.fval1];[Mboot.fval3]]';
cfg.nperm = [10000];
compare_mod1_mod3 = kat_stats_perm_maxT(cfg);

% Compare model 2 with model 3
cfg       =[];
cfg.type  = 'two-sample-paired-ttest';%one-sample two-sample-paired
cfg.data  = [[Mboot.fval2];[Mboot.fval3]]';
cfg.nperm = [10000];
compare_mod2_mod3 = kat_stats_perm_maxT(cfg);

%EXTRACTING WEIGHTS TO CALCULATE SCORES AND LOADINGS
%ORIGINALLLY SET FOR THE WINNING MODEL
%IMPLEMENTED FOR ALL MODELS TO PLOT THE LOADINGS CHART



% MODEL 1 - WHEIGHTS, SCORES AND LOADINGS

% (extrating weights)
nvarX_model1 = size(M{1}.X,2);
nvarY_model1 = size(M{1}.Y,2);
XW_model1    = reshape([Mboot.XW1],nvarX_model1,numboot);
YW_model1   = reshape([Mboot.YW1],nvarY_model1,numboot);
XL_model1  = reshape([Mboot.XL1],nvarX_model1,numboot);
YL_model1  = reshape([Mboot.YL1],nvarY_model1,numboot);



% Realign Weights in the same direction
for i = 2:numboot
    if (XW_model1(:,1)' * XW_model1(:,i))<0
        XW_model1(:,i)  = -XW_model1(:,i);
        YW_model1(:,i)  = -YW_model1(:,i);
        XL_model1(:,i)  = -XL_model1(:,i);
        YL_model1(:,i)  = -YL_model1(:,i);
    end
end


%compute the scores based on the weights
XS_model1 = M{1}.X * median(XW_model1,2);
YS_model1 = M{1}.Y * median(YW_model1,2);
XL_model1 = corr(M{1}.X,XS_model1);
YL_model1 = corr(M{1}.Y,YS_model1);


%-------


% MODEL 2 - WHEIGHTS, SCORES AND LOADINGS

% (extrating weights)
nvarX_model2 = size(M{2}.X,2);
nvarY_model2 = size(M{2}.Y,2);
XW_model2    = reshape([Mboot.XW2],nvarX_model2,numboot);
YW_model2   = reshape([Mboot.YW2],nvarY_model2,numboot);
XL_model2  = reshape([Mboot.XL2],nvarX_model2,numboot);
YL_model2  = reshape([Mboot.YL2],nvarY_model2,numboot);


% Realign Weights in the same direction
for i = 2:numboot
    if (XW_model2(:,1)' * XW_model2(:,i))<0
        XW_model2(:,i)  = -XW_model2(:,i);
        YW_model2(:,i)  = -YW_model2(:,i);
        XL_model2(:,i)  = -XL_model2(:,i);
        YL_model2(:,i)  = -YL_model2(:,i);
    end
end

%compute the scores based on the weights
XS_model2 = M{2}.X * median(XW_model2,2);
YS_model2 = M{2}.Y * median(YW_model2,2);
XL_model2 = corr(M{2}.X,XS_model2);
YL_model2 = corr(M{2}.Y,YS_model2);



%------

% MODEL 3 - WHEIGHTS, SCORES AND LOADINGS

% (extrating weights)
nvarX_model3 = size(M{3}.X,2);
nvarY_model3 = size(M{3}.Y,2);
XW_model3    = reshape([Mboot.XW3],nvarX_model3,numboot);
YW_model3   = reshape([Mboot.YW3],nvarY_model3,numboot);
XL_model3  = reshape([Mboot.XL3],nvarX_model3,numboot);
YL_model3  = reshape([Mboot.YL3],nvarY_model3,numboot);


% Realign Weights in the same direction
for i = 2:numboot
    if (XW_model3(:,1)' * XW_model3(:,i))<0
        XW_model3(:,i)  = -XW_model3(:,i);
        YW_model3(:,i)  = -YW_model3(:,i);
        XL_model3(:,i)  = -XL_model3(:,i);
        YL_model3(:,i)  = -YL_model3(:,i);
    end
end

%compute the scores based on the weights
XS_model3 = M{3}.X * median(XW_model3,2);
YS_model3 = M{3}.Y * median(YW_model3,2);
XL_model3 = corr(M{3}.X,XS_model3);
YL_model3 = corr(M{3}.Y,YS_model3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PLOTTING LOADINGS

% Let's plot the loadings for the  for the first canonical variate (aka
% component), the only significan canonical variate in your analysis

% MODEL 1

figure;
subplot(1,2,1);
bar(XL_model1(:,1));
subplot(1,2,2);
bar(YL_model1(:,1));

XL_model1INV = XL_model1*-1;
YL_model1INV = YL_model1*-1; 
figure;
subplot(1,2,1);
bar(XL_model1INV(:,1));
subplot(1,2,2);
bar(YL_model1INV(:,1));

% MODEL 2
figure;
subplot(1,2,1);
bar(XL_model2(:,1));
subplot(1,2,2);
bar(YL_model2(:,1));

XL_model2INV = XL_model2*-1;
YL_model2INV = YL_model2*-1; 
figure;
subplot(1,2,1);
bar(XL_model2INV(:,1));
subplot(1,2,2);
bar(YL_model2INV(:,1));


% MODEL 3
figure;
subplot(1,2,1);
bar(XL_model3(:,1));
subplot(1,2,2);
bar(YL_model3(:,1));


XL_model3INV = XL_model3*-1;
YL_model3INV = YL_model3*-1; 
figure;
subplot(1,2,1);
bar(XL_model3INV(:,1));
subplot(1,2,2);
bar(YL_model3INV(:,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Get significance of Loadings for second model 
% based on repeating boostraps with permuted rows in Y
imodel = 2;
X = M{imodel}.X;
Y = M{imodel}.Y;

numPerm = 1000;
nsub    = size(Y,1);
XLnull  = nan(nvarX_model2,numPerm);
YLnull  = nan(nvarY_model2,numPerm);
parfor iperm = 1:numPerm
    
    tempOrder = randperm(nsub);
    Ytemp     = Y(tempOrder,:);

    % Get the loadings for the best performing bootstrap
    Mperm = bootstrp(numboot, @myccabootstr_singleModel,X,Ytemp); % < -- This works
    [~,i] = max([Mperm.fval1]); % keep fval1 for all models
    XLnull(:,iperm) = Mperm(i).XL1;% keep L1 for all models
    YLnull(:,iperm) = Mperm(i).YL1; % keep L1 for all models
end
 
% ------------------------------------------------------------
% Determine significance based on permutations
% ------------------------------------------------------------
pPermXL_model2 = sum(abs(XLnull) > repmat(abs(median(XL_model2,2)),1,numPerm),2)/numPerm;
pPermYL_model2 = sum(abs(YLnull) > repmat(abs(median(YL_model2,2)),1,numPerm),2)/numPerm;



%%%%%%

%Get significance of Loadings for THE FIRST model 
% based on repeating boostraps with permuted rows in Y
imodel = 1;
X = M{imodel}.X;
Y = M{imodel}.Y;

numPerm = 1000;
nsub    = size(Y,1);
XLnull  = nan(nvarX_model1,numPerm);
YLnull  = nan(nvarY_model1,numPerm);
parfor iperm = 1:numPerm
    
    tempOrder = randperm(nsub);
    Ytemp     = Y(tempOrder,:);

    % Get the loadings for the best performing bootstrap
    Mperm = bootstrp(numboot, @myccabootstr_singleModel,X,Ytemp); % < -- This works
    [~,i] = max([Mperm.fval1]); % keep fval1 for all models
    XLnull(:,iperm) = Mperm(i).XL1;% keep L1 for all models
    YLnull(:,iperm) = Mperm(i).YL1; % keep L1 for all models
end
 
% ------------------------------------------------------------
% Determine significance based on permutations
% ------------------------------------------------------------
pPermXL_model1 = sum(abs(XLnull) > repmat(abs(median(XL_model1,2)),1,numPerm),2)/numPerm;
pPermYL_model1 = sum(abs(YLnull) > repmat(abs(median(YL_model1,2)),1,numPerm),2)/numPerm;


%%%%%%

%Get significance of Loadings for THE THIRD model 
% based on repeating boostraps with permuted rows in Y
imodel = 3;
X = M{imodel}.X;
Y = M{imodel}.Y;

numPerm = 1000;
nsub    = size(Y,1);
XLnull  = nan(nvarX_model3,numPerm);
YLnull  = nan(nvarY_model3,numPerm);
parfor iperm = 1:numPerm
    
    tempOrder = randperm(nsub);
    Ytemp     = Y(tempOrder,:);

    % Get the loadings for the best performing bootstrap
    Mperm = bootstrp(numboot, @myccabootstr_singleModel,X,Ytemp); % < -- This works
    [~,i] = max([Mperm.fval1]); % keep fval1 for all models
    XLnull(:,iperm) = Mperm(i).XL1;% keep L1 for all models
    YLnull(:,iperm) = Mperm(i).YL1; % keep L1 for all models
end
 
% ------------------------------------------------------------
% Determine significance based on permutations
% ------------------------------------------------------------
pPermXL_model3 = sum(abs(XLnull) > repmat(abs(median(XL_model3,2)),1,numPerm),2)/numPerm;
pPermYL_model3 = sum(abs(YLnull) > repmat(abs(median(YL_model3,2)),1,numPerm),2)/numPerm;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%

% Correlations 


corr_model1 = corr(XS_model1,YS_model1)
[corr_model1R,corr_model1P] = corrcoef(XS_model1,YS_model1)

corr_model2 = corr(XS_model2,YS_model2)
[corr_model2R,corr_model2P] = corrcoef(XS_model2,YS_model2)


corr_model3 = corr(XS_model3,YS_model3)
[corr_model3R,corr_model3P] = corrcoef(XS_model3,YS_model3)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% REGRESSION (ONLY FOR THE WINNING MODEL)
T = table();
T.age = zage;
T.gender = zgender;
T.quali = zquali;
T.xs_boot_model2 = -XS_model2;
T.ys_boot_model2 = -YS_model2;

%T.quali_cat = categorical(T.quali);

%model = 'ys_csa ~ xs_csa + age';
%model = 'ys_boot ~ xs_boot + age + quali';


%model_age = 'ys_boot_model2 ~ xs_boot_model2 + age + age*xs_boot_model2';
%mlr_age = fitlm(T,model_age)

%model_gender = 'ys_boot_model2 ~ xs_boot_model2 + age + gender + gender*xs_boot_model2';
%mlr_gender = fitlm(T,model_gender)

model_age = 'ys_boot_model2 ~ xs_boot_model2 + age + quali + age*xs_boot_model2';
mlr_age = fitlm(T,model_age)

model_gender = 'ys_boot_model2 ~ xs_boot_model2 + age + quali + gender + gender*xs_boot_model2';
mlr_gender = fitlm(T,model_gender)

model_age_gender = 'ys_boot_model2 ~ xs_boot_model2 + age + quali + gender + age*xs_boot_model2 + gender*xs_boot_model2';
mlr_age_gender = fitlm(T,model_age_gender)

model_age_gender_3int = 'ys_boot_model2 ~ xs_boot_model2 + age + quali + gender + age*xs_boot_model2 + gender*xs_boot_model2 + age*xs_boot_model2*gender';
mlr_age_gender = fitlm(T,model_age_gender_3int)






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% ---------------------------------
% 1 Set up Canonical Spectal Analysis
% ---------------------------------
%[coeff, score, latent, tsquared, explained] = pca(zscore(M{1}.Y));
[coeff, score, latent, tsquared, explained] = pca(M{1}.Y);
for imodel = 1:3
    CCA     = [];
    %CCA.X   = zscore(M{imodel}.X);
    %CCA.Y   = zscore((M{1}.Y));%(score(:,1));
    CCA.X   = M{imodel}.X;
    CCA.Y   = M{1}.Y;%(score(:,1));

    CCA.mode.cv.do              = 1; % Cross-validation settings
    CCA.mode.cv.numFolds        = 5;
    CCA.mode.cv.numPart         = 5000; % How many time to repeat CV with different patitions
    CCA.mode.cv.permutePW       = 1; % Partition-wise permutations
    CCA.mode.cv.numPermPW       = 5000;
    CCA.mode.cv.permuteFW       = 0;
    CCA.mode.cv.numPermFW       = 100;
    CCA.mode.cv.doSplitHalfDW   = 0;
    CCA.mode.cv.doSplitHalfFW   = 0;
    CCA.mode.cv.numBootDW       = 1000;
    CCA.mode.cv.doSaveNullsFW   = 0;
    CCA.mode.cv.doSaveNullsDW   = 0;
    CCA.mode.cv.labels          = ceil((age-7)./10)-1;

    CCA.mode.permClassic.do     = 0;        % Classical Permutations
    CCA.mode.permClassic.numPerm= 5000;

    CCA.numComp             = min([...10,...
                                  size(CCA.X,2)...
                                  size(CCA.Y,2)]);

    CCA.lambdaX             = 0;           % 'auto' or [0-1], where 0 is CCA, 1 is PLS, 0-1 is regularized CCA
    CCA.lambdaY             = 0; % 'auto' or [0-1], where 0 is CCA, 1 is PLS, 0-1 is regularized CCA
    CCA.numComp             = size(CCA.Y,2);
    CCA.doSaveNulls         = 1;
    CCA.usePresetRandOrder  = 0;
    CCA.nameAnalysis        = 'prelim';
    CCA.dirRoot             = path2results;

    tic, [CCA,CCApart,CCAnull] = csa_stats_rCVA_wrapper(CCA); toc
    M{imodel}.cca     = CCA;
    M{imodel}.CCApart = CCApart;
    M{imodel}.CCAnull = CCAnull;
end


%% Compare models with different complexity 
% -----------------------------------------
whichCV     = 1;
whichStat   = 'Rdw';
rval1 = [M{1}.CCApart.(whichStat)]'; rval1 = rval1(:,whichCV);
rval2 = [M{2}.CCApart.(whichStat)]'; rval2 = rval2(:,whichCV);
rval3 = [M{3}.CCApart.(whichStat)]'; rval3 = rval3(:,whichCV);
null1 = [M{1}.CCAnull.(whichStat)]'; null1 = null1(:,whichCV);
null2 = [M{2}.CCAnull.(whichStat)]'; null2 = null2(:,whichCV);
null3 = [M{3}.CCAnull.(whichStat)]'; null3 = null3(:,whichCV);


% -----------------------
% Visualise distributions
% -----------------------
figure('position',[30 300 1200 400])
ranges = [0.2:.001:0.7];
cols   = cbrewer('qual','Set1',3);
histogram(rval1,ranges,'facecolor',cols(2,:),'facealpha',.5,'edgecolor','none')
hold on
histogram(rval2,ranges,'facecolor',cols(1,:),'facealpha',.5,'edgecolor','none') 
histogram(rval3,ranges,'facecolor',cols(3,:),'facealpha',.5,'edgecolor','none') 
legend model1 model2 model3
title(['Lambda ' num2str(M{1}.cca.lambdaX)]);
xlabel('r-value');
ylabel('frequencies');


figure('position',[30 300 1200 400]);
ranges = [-0.5:.001:0.5];
histogram(null1,ranges,'facecolor',cols(2,:),'facealpha',.5,'edgecolor','none')
hold on
histogram(null2,ranges,'facecolor',cols(1,:),'facealpha',.5,'edgecolor','none') 
histogram(null3,ranges,'facecolor',cols(3,:),'facealpha',.5,'edgecolor','none') 
legend null1 null2 null3
title(['Lambda ' num2str(M{1}.cca.lambdaX)]);
xlabel('r-value');
ylabel('frequencies');

% -------------------------------------
% Compare Partitionining distribituions
% -------------------------------------
cfg       =[];
cfg.type  = 'one-sample';
cfg.data  = [rval1,rval2,rval3];
cfg.nperm = 1000;
cfg       = kat_stats_perm_maxT(cfg);
tval      = cfg.tstat
pvals     = cfg.p_orig
pPerm     = cfg.p_perm

% -------------------------------------
% Do some work with Loadings
% valXLpval returns p-values for datset X Loadings, based on Null loadings in CCAnull
% valYLpval returns p-values for datset Y Loadings, based on Null loadings in CCAnull
% -------------------------------------
%varnameLoading = {'valXL', 'valYL'};
%for imodel = 1:3
%    for ivar = 1:numel(varnameLoading)
%        varname = varnameLoading{ivar};
%        L = arrayfun(@(x) x.(varname)(:,1),M{imodel}.CCApart,'UniformOutput',0); % Unpack loadings for dataset X
%        L = [L{:}];
%        [coeff, score, latent, tsquared, explained] = pca(L); % Run PCA to realign arbitrary sign assignment in cca
%        L = L .* sign(score(:,1)); 
%        Lnull = arrayfun(@(x) x.(varname)(:,1),M{imodel}.CCAnull,'UniformOutput',0);
%        Lnull = abs([Lnull{:}]);
%       numperm = size(Lnull,2);
%        M{imodel}.([varname 'pval']) = sum(Lnull > repmat(median(L,2),1,numperm),2)./numperm;
%    end
%end

varnameLoading = {'valXL', 'valYL'};
for imodel = 1:3
    for ivar = 1:numel(varnameLoading)
        varname = varnameLoading{ivar};
        L = arrayfun(@(x) x.(varname)(:,1),M{imodel}.CCApart,'UniformOutput',0); % Unpack loadings for dataset X
        L = [L{:}]';
        [coeff, score, latent, tsquared, explained] = pca(L,'Centered',false); % Run PCA to realign arbitrary sign assignment in cca      
        L = L .* sign(score(:,1));
        Lnull = arrayfun(@(x) x.(varname)(:,1),M{imodel}.CCAnull,'UniformOutput',0);
        Lnull = abs([Lnull{:}]);
        numperm = size(Lnull,2);
        M{imodel}.([varname 'pval']) = sum(Lnull > repmat(abs(median(L',2)),1,numperm),2)./numperm;
    end
end

%% CCA BOOTSTRAP
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------%%
% Y = score(:,1);t
numboot = 5000;
% M1 = bootstrp(numboot, @myccabootstr,M{1}.X,M{1}.Y); % < -- This works
% M2 = bootstrp(numboot, @myccabootstr,M{2}.X,M{2}.Y); % < -- This works
% M3 = bootstrp(numboot, @myccabootstr,M{3}.X,M{3}.Y); % < -- This works

Mboot = bootstrp(numboot, @myccabootstr_cmp,M{1}.X,M{2}.X,M{3}.X,M{1}.Y); % < -- This works
figure;histogram([Mboot.fval1]);hold on;histogram([Mboot.fval2]);histogram([Mboot.fval3])


% Compare F values (Update variables)
% -------------------------------------
% Compare Partitionining distribituions
% -------------------------------------
cfg       =[];
cfg.type  = 'one-sample';
cfg.data  = [Mboot.fval1,Mboot.fval2,Mboot.fval3];
cfg.nperm = 1000;
cfg       = kat_stats_perm_maxT(cfg);
tval      = cfg.tstat
pvals     = cfg.p_orig
pPerm     = cfg.p_perm


%For the winning model (extrating weights)

nvarX = size(M{2}.X,2);
nvarY = size(M{2}.Y,2);
XW    = reshape([Mboot.XW2],nvarX,numboot);
YW    = reshape([Mboot.YW2],nvarY,numboot);


% Realign Weights in the same direction
for i = 2:numboot
    if (XW(:,1)' * XW(:,i))<0
        XW(:,i)  = -XW(:,i);
        YW(:,i)  = -YW(:,i);
    end
end

%compute the scores based on the weights
XS = M{2}.X * median(XW,2);
YS = M{2}.Y * median(YW,2);
XL = corr(M{2}.X,XS);
T.xs_boot = XS;
T.ys_boot = YS;

% model = 'ys_csa ~ xs_csa + age';
model = 'ys_boot ~ xs_boot + age';
mlr = fitlm(T,model)


%M2vs1 = bootstrp(numboot, @myccabootstr_cmp,M{1}.X,M{2}.X,M{1}.Y); % < -- This works
%mean([M2vs1.fval2]-[M2vs1.fval1]);% Differences in F-stats
% nvarX1 = size(M{1}.X,2);
% nvarX2 = size(M{2}.X,2);
% nvarY1 = size(M{1}.Y,2);
% nvarY2 = size(M{2}.Y,2);
% nsub   = size(M{1}.Y,1);
% XL1    = reshape([M2vs1.XL1],nvarX1,numboot)';
% XL2    = reshape([M2vs1.XL2],nvarX2,numboot)';
% YL1    = reshape([M2vs1.YL1],nvarY1,numboot)';
% YL2    = reshape([M2vs1.YL2],nvarY2,numboot)';
% % Run PCA to realign arbitrary sign assignment in cca
% [coeff, score, latent, tsquared, explained] = pca(XL1);
% XL1 = XL1 .* sign(score(:,1));
% [coeff, score, latent, tsquared, explained] = pca(XL2);
% XL2 = XL2 .* sign(score(:,1));
% [coeff, score, latent, tsquared, explained] = pca(YL1);
% YL1 = YL1 .* sign(score(:,1));
% [coeff, score, latent, tsquared, explained] = pca(YL2);
% YL2 = YL2 .* sign(score(:,1));

M2vs1 = bootstrp(numboot, @myccabootstr_cmp,M{1}.X,M{2}.X,M{1}.Y); % < -- This works
mean([M2vs1.fval2]-[M2vs1.fval1])
figure;histogram([M2vs1.fval2]);hold on;histogram([M2vs1.fval1])

Mboot = bootstrp(numboot, @myccabootstr_cmp,M{1}.X,M{2}.X,M{3}.X,M{1}.Y); % < -- This works
figure;histogram([Mboot.fval1]);hold on;histogram([Mboot.fval2]);histogram([Mboot.fval3])

mean([M3vs2.fval2]-[M3vs2.fval1])

%% MLR comparison
[coeff, score, latent, tsquared, explained] = pca(zscore(M{1}.Y));
[Lambda, Psi, T, stats, F] = factoran(zscore(M{1}.Y), 1, 'scores', 'regr')

Y = score(:,1);
for imodel = 1:3
    M{imodel}.mlr = fitlm(zscore(M{imodel}.X),Y);
end

% % Compare Model2 to Model1
% % Probably not appropriate as LogLikelihood requires same complexity model
% M2vsM1 = 2*(M{2}.mlr.LogLikelihood - M{1}.mlr.LogLikelihood); % has a X2 distribution with a df equals to number of constrained parameters, here: 1
% pval = 1 - chi2cdf(M1vsM2, 1);
% 
% % Compare Model3 to Model2
% M3vsM2 = 2*(M{3}.mlr.LogLikelihood - M{2}.mlr.LogLikelihood); % has a X2 distribution with a df equals to number of constrained parameters, here: 1
% pval = 1 - chi2cdf(M3vsM2, 1);

%% Bootstrapping
Y = score(:,1);

% Parameters significance/bootstrap
for imodel = 1:3
    x = zscore(M{imodel}.X);
    x = [ones(size(x,1),1) x];
    b = regress(Y,x);
    yfit = x*b;
    resid = Y - yfit;
    se = std(bootstrp(10000,@(bootr)regress(yfit+bootr,x),resid));

    LL{imodel} = bootstrp(1000, @myfitlm, x, Y); % < -- This works
end

%% MLR model-fit R adjustes bootstrap
Y = score(:,1);
M2vsM1 = bootstrp(1000, @mymodelcompare,zscore(M{1}.X),zscore(M{2}.X), Y); % Compare Model2 vs Model1
M3vsM2 = bootstrp(1000, @mymodelcompare,zscore(M{2}.X),zscore(M{3}.X), Y); % Compare Model3 vs Model2
M3vsM1 = bootstrp(1000, @mymodelcompare,zscore(M{1}.X),zscore(M{3}.X), Y); % Compare Model3 vs Model1

% myout = mean(1 - chi2cdf(M3vsM2, 1));
figure('position',[30 300 1200 400])
ranges = [-0.1:.005:0.3];
histogram(M2vsM1,ranges,'facecolor',cols(2,:),'facealpha',.5,'edgecolor','none')
hold on
histogram(M3vsM2,ranges,'facecolor',cols(1,:),'facealpha',.5,'edgecolor','none') 
histogram(M3vsM1,ranges,'facecolor',cols(3,:),'facealpha',.5,'edgecolor','none') 
legend M2vsM1 M3vsM2 M3vsM1
title('MLR R2-adjusted bootstrap');
xlabel('Difference in R2-adjusted');
ylabel('frequencies');




