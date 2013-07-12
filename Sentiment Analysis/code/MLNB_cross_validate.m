function [error rmse] = MLNB_cross_validate(X,Y,n_folds)

Yk = bsxfun(@eq, Y, [1 2 4 5]);
Yk(Yk==0) = -1;

parts = make_xval_partition(size(X,1),n_folds);

n_folds = max(parts);
part_errors = zeros(n_folds,1);
part_rmse = zeros(n_folds,1);

for part = 1:n_folds
    part_mask = parts==part;
    
    [Prior,PriorN,mu,muN,sigma,sigmaN] = MLNB_Basic_train(X(~part_mask,:),Yk(~part_mask,:)',0.1);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB_Basic_test(X(part_mask,:),Yk(part_mask,:)',Prior,PriorN,mu,muN,sigma,sigmaN);
    Y_tilde = Outputs';
        
    Y_tilde = sum(bsxfun(@times,Y_tilde,[1 2 4 5]),2);
        
    %Calculating error
    Y_tilde_rounded = round(Y_tilde);    
    part_errors(part) = mean(Y_tilde_rounded ~= Y(part_mask));
    
    %Calculating rmse
    part_rmse(part) = sqrt(mean((Y(part_mask) - Y_tilde).^2));
    
    disp('Finished a fold');
end

error = mean(part_errors)
rmse = mean(part_rmse)


end

