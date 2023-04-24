function myout = myfitlm(preds,tobepred)
    % select out the coefficients that are to be bootstrapped.
    b21 = fitlm(preds,tobepred);
%     myout = b21.Coefficients.Estimate;
    myout = b21.LogLikelihood;
end