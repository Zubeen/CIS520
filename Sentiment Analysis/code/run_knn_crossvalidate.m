% sample training data randomly
random = make_xval_partition(size(X_combined_reduced5_shortened1,1),620);
random_mask = random==5;

% divide randomly sampled data into random parts
parts = make_xval_partition(size (X_combined_reduced5_shortened1(~),1),5);

% run knn_cross_validate on randomly sampled data
knn_cross_validate2(X_combined_reduced5_shortened1, Y_shortened51,parts,random_mask)

