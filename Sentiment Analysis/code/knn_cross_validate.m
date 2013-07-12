function [error rmse] = knn_cross_validate(X, Y,nfolds)

parts = make_xval_partition(size(X,1),nfolds);

n_folds = max(parts);
part_errors = zeros(n_folds,1);
part_rmse = zeros(n_folds,1);

%t = CTimeleft(size(X, 1));
for part = 1:n_folds
    %t.timeleft();
    part_mask = parts==part;
    
    Y_tilde = knnclassify(X(part_mask,:),X(~part_mask,:),Y(~part_mask),10,'euclidean','nearest');
      
    %If -b 1 flag is set, comment these 2 lines below
%     Y_tilde = exp(Y_tilde);
%     Y_tilde = bsxfun(@times, Y_tilde, 1./sum(Y_tilde,2));
%     
%     Y_tilde = sum(bsxfun(@times,Y_tilde,[1 2 4 5]),2);
%         
%     %Calculating error
%     Y_tilde_rounded = round(Y_tilde);    
    part_errors(part) = mean(Y_tilde ~= Y(part_mask));
    
    %Calculating rmse
%     Y_tilde (find(Y_tilde > 2 & Y_tilde < 2.75))=2;
%     Y_tilde (find(Y_tilde > 3.25 & Y_tilde < 4))=4;
    part_rmse(part) = sqrt(mean((Y(part_mask) - Y_tilde).^2));
    
    disp('Finished a fold');
end

error = mean(part_errors)
rmse = mean(part_rmse)
