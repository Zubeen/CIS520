function [error rmse] = nb_cross_validate(X, Y,nfolds)

parts = make_xval_partition(size(X,1),nfolds);

n_folds = max(parts);
part_errors = zeros(n_folds,1);
part_rmse = zeros(n_folds,1);

Yk = bsxfun(@eq, Y, [1 2 4 5]);

for part = 1:n_folds
    part_mask = parts==part;
    
    nb = nb_train_pk(X(~part_mask,:)'>0,Yk(~part_mask,:));
    Y_tilde = nb_test_pk(nb,X(part_mask,:)'>0);
    
    %[tmp, Y_tilde_argmax] = max(Y_tilde, [], 2);
    
    Y_tilde = sum(bsxfun(@times,Y_tilde,[1 2 4 5]),2);
    
    %Y_tilde = zeros(size(part_mask),1) + 5;    
    
    %Minor hacks
    %ratings = [1 2 4 5];
    %Y_tilde_argmax = ratings(Y_tilde_argmax);
    %Y_tilde(Y_tilde > 2 & Y_tilde < 2.5) = 2;
    %Y_tilde(Y_tilde >3.5 & Y_tilde < 4) = 4;
    %Y_tilde(Y_tilde >= 2.5 & Y_tilde <= 3.5) = Y_tilde_argmax(Y_tilde >= 2.5 & Y_tilde <= 3.5);
    
    %Calculating error
    Y_tilde_rounded = round(Y_tilde);    
    part_errors(part) = mean(Y_tilde_rounded ~= Y(part_mask));
    
    %Calculating rmse
    part_rmse(part) = sqrt(mean((Y(part_mask) - Y_tilde).^2));
    
    disp('Finished a fold');
end

error = mean(part_errors)
rmse = mean(part_rmse)