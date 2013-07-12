%% Initialize
addpath liblinear;
load ../data/data_with_bigrams.mat;

%% Preprocessing - clean Unigrams
[cleaned_unigram_vocab new_unigram_indices] = clean_vocab(vocab);
[cleaned_unigram_train] = clean_data(train,new_unigram_indices);
cleaned_unigram_test = clean_data(test,new_unigram_indices);

%% Preprocessing - clean Bigrams

% Cleaning the vocabulary and changing the word_idx
[cleaned_bigram_vocab new_indices] = clean_bigram_vocab(bigram_vocab);
cleaned_bigram_train = clean_bigram_data(train,new_indices);
cleaned_bigram_test = clean_bigram_data(test,new_indices);

%% Preprocessing - 2 Helpful Feature
% % Using the helpful feature to get weighted reviews - Don't use, not
% getting good results
% [cleaned_bigram_train_helpful helpful_train] = use_helpful_feature(cleaned_bigram_train,0);
% [cleaned_bigram_test_helpful helpful_test] = use_helpful_feature(cleaned_bigram_test,0);

%% Preprocessing -3 Creating the sparse matrices

X = make_sparse(cleaned_unigram_train);
X_bigram = make_bigram_sparse(cleaned_bigram_train);
Y = double([cleaned_bigram_train.rating]');

Xtest= make_sparse(cleaned_unigram_test,size(X,2));
Xtest_bigram = make_bigram_sparse(cleaned_bigram_test,size(X_bigram,2));

%% Reducing the number of features - Initial Reduction

[X_reduced  new_id]= reduce_features(X,0);
[X_bigram_reduced new_big_id] = reduce_features(X_bigram,0);
X_combined = [X_reduced X_bigram_reduced];
Xtest_combined = [Xtest(:,new_id) Xtest_bigram(:,new_big_id)];

%% Reduce Features on the basis of weights assigned by liblinear

lib_classifier5 = liblinear_train(Y,X_combined,'-c 0.5 -s 5 -e 1.0', 'row');
count=0;
for i=1:size(X_combined,2)
    if(lib_classifier5.w(1,i)==0 && lib_classifier5.w(2,i)==0 && lib_classifier5.w(3,i)==0 && lib_classifier5.w(4,i)==0 )
    else
        count=count+1;
        X_combined_reduced5(:,count)=X_combined(:,i);
        Xtest_combined_reduced5(:,count)=Xtest_combined(:,i);
    end
end    

%% Remove Outliers by Predicting on Training Data
lib_classifier7_reduced5 = liblinear_train(Y,X_combined_reduced5,'-c 0.25 -s 7 -e 1.0', 'row')
[label accuracy Yhattemp] = liblinear_predict(ones(size(X_combined_reduced5,1),1), X_combined_reduced5, lib_classifier7_reduced5, '-b 0','row');

Yhattemp = exp(Yhattemp);
Yhattemp = bsxfun(@times, Yhattemp, 1./sum(Yhattemp,2));
Yhattemp = sum(bsxfun(@times,Yhattemp,[1 2 4 5]),2);

X_combined_reduced5_shortened1= X_combined_reduced5(find (abs(Yhattemp - Y) < 1 ),:);
Y_shortened51= Y(find (abs(Yhattemp - Y) < 1 ),:);

%% Final Training and Prediction
%Training
liblinear_logistic_classifier = liblinear_train(Y_shortened51,X_combined_reduced5_shortened1,'-c 0.25 -s 7 -e 1.0 ', 'row');

%Testing
[predicted_label, accuracy, Yhat] = liblinear_predict(ones(size(Xtest_combined_reduced5,1),1), Xtest_combined_reduced5, liblinear_logistic_classifier, '-b 0','row');

Yhat = exp(Yhat);
Yhat = bsxfun(@times, Yhat, 1./sum(Yhat,2));
Yhat = sum(bsxfun(@times,Yhat,[1 2 4 5]),2);
%% Saving results
save('-ascii', 'submit.txt', 'Yhat');


%%




