%% Reducing the number of features - Initial Reduction

[X_reduced_sw  new_id_sw]= reduce_features(X_sw,0);
X_combined_sw = [X_reduced_sw X_bigram_reduced];


%% Reduce Features on the basis of weights assigned by liblinear

lib_classifier5 = liblinear_train(Y,X_combined_sw,'-c 0.5 -s 5 -e 1.0', 'row');
count=0;
for i=1:size(X_combined_sw,2)
    if(lib_classifier5.w(1,i)==0 && lib_classifier5.w(2,i)==0 && lib_classifier5.w(3,i)==0 && lib_classifier5.w(4,i)==0 )
    else
        count=count+1;
        X_combined_reduced5_sw(:,count)=X_combined_sw(:,i);
    end
end    

%% Remove Outliers by Predicting on Training Data
lib_classifier7_reduced5 = liblinear_train(Y,X_combined_reduced5_sw,'-c 0.25 -s 7 -e 1.0', 'row')
[label accuracy Yhattemp] = liblinear_predict(ones(size(X_combined_reduced5_sw,1),1), X_combined_reduced5_sw, lib_classifier7_reduced5, '-b 0','row');

Yhattemp = exp(Yhattemp);
Yhattemp = bsxfun(@times, Yhattemp, 1./sum(Yhattemp,2));
Yhattemp = sum(bsxfun(@times,Yhattemp,[1 2 4 5]),2);

X_combined_reduced5_shortened1_sw= X_combined_reduced5(find (abs(Yhattemp - Y) < 1 ),:);
Y_shortened51_sw= Y(find (abs(Yhattemp - Y) < 1 ),:);
