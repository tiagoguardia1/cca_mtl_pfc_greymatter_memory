function [cfgOut] = myccabootstr(x1,x2,x3,y)
    % select out the coefficients that are to be bootstrapped.
    x1 = zscore(x1);
    x2 = zscore(x2);
    x3 = zscore(x3);
    y  = zscore(y);

    [A1,B1,R1,U1,V1,STATS1] = canoncorr(x1,y);

    [A2,B2,R2,U2,V2,STATS2] = canoncorr(x2,y);
    
     [A3,B3,R3,U3,V3,STATS3] = canoncorr(x3,y);
%     M2vsM1 = STATS2.F - STATS1.F;
    
    cfgOut.fval1= STATS1.F(1);
    cfgOut.fval2= STATS2.F(1);
    cfgOut.fval3= STATS3.F(1);
%    cfgOut.fval2= STATS2.F(1);
%     cfgOut.XL1  = corr(U1(:,1),x1);
%     cfgOut.XL2  = corr(U2(:,1),x2);
%     cfgOut.YL1  = corr(V1(:,1),y);
%     cfgOut.YL2  = corr(V2(:,1),y);
% %     cfgOut.A1  = A1(:,1);
%     cfgOut.B1  = B1(:,1);
%     cfgOut.A2  = A2(:,1);
%     cfgOut.B2  = B2(:,1);

    % output weigths
    cfgOut.XW1  = A1(:,1);
    cfgOut.YW1  = B1(:,1);
    cfgOut.XW2  = A2(:,1);
    cfgOut.YW2  = B2(:,1);
    cfgOut.XW3  = A3(:,1);
    cfgOut.YW3  = B3(:,1);

end
