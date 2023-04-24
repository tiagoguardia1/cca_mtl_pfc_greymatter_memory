function [cfgOut] = myccabootstr(x1,y)
    % select out the coefficients that are to be bootstrapped.
    x1 = zscore(x1);
    y  = zscore(y);

    [A1,B1,R1,U1,V1,STATS1] = canoncorr(x1,y);

%     [A2,B2,R2,U2,V2,STATS2] = canoncorr(x2,y);
%     M2vsM1 = STATS2.F - STATS1.F;
    
    cfgOut.fval= STATS1.F(1);
%     cfgOut.fval2= STATS2.F(1);
    cfgOut.XL1  = corr(U1(:,1),x1);
%     cfgOut.XL2  = corr(U2(:,1),x2);
    cfgOut.YL1  = corr(V1(:,1),y);
%     cfgOut.YL2  = corr(V2(:,1),y);
    cfgOut.XW  = A1(:,1);
    cfgOut.YW  = B1(:,1);
%     cfgOut.A2  = A2(:,1);
%     cfgOut.B2  = B2(:,1);
    
end
