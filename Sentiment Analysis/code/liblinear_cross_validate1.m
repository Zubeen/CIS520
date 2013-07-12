function [error rmse] = liblinear_cross_validate1(X, Y,parts)

% parts = make_xval_partition(size(X,1),nfolds);

n_folds = max(parts);
part_errors = zeros(n_folds,1);
part_rmse = zeros(n_folds,1);

%t = CTimeleft(size(X, 1));
for part = 1:n_folds
    %t.timeleft();
    part_mask = parts==part;
    
    lib_classifier = liblinear_train(Y(~part_mask),X(~part_mask,:),'-c 0.25 -s 7 -e 1.0', 'row');
    [label accuracy Y_tilde] = liblinear_predict(Y(part_mask), X(part_mask,:), lib_classifier, '-b 0','row');
      
    %If -b 1 flag is set, comment these 2 lines below
    Y_tilde = exp(Y_tilde);
    Y_tilde = bsxfun(@times, Y_tilde, 1./sum(Y_tilde,2));
    
    Y_tilde = sum(bsxfun(@times,Y_tilde,[1 2 4 5]),2);
        
    %Calculating error
    Y_tilde_rounded = round(Y_tilde);    
    part_errors(part) = mean(Y_tilde_rounded ~= Y(part_mask));
    
    %Calculating rmse
%     Y_tilde (find(Y_tilde > 2 & Y_tilde < 2.5))=2;
%     Y_tilde (find(Y_tilde > 3.5 & Y_tilde < 4))=4;
    part_rmse(part) = sqrt(mean((Y(part_mask) - Y_tilde).^2));
    
    disp('Finished a f2old');
end

error = mean(part_errors)
rmse = mean(part_rmse)
