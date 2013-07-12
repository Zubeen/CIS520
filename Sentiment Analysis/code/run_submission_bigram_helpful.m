addpath liblinear;
% load ../data/data_with_bigrams.mat;

% Cleaning the vocabulary and changing the word_idx
[cleaned_bigram_vocab new_indices] = clean_bigram_vocab(bigram_vocab);
cleaned_bigram_train = clean_bigram_data(train,new_indices);
cleaned_bigram_test = clean_bigram_data(test,new_indices);

% % Using the helpful feature to get weighted reviews
[cleaned_bigram_train_helpful helpful_train] = use_helpful_feature(cleaned_bigram_train,0);
[cleaned_bigram_test_helpful helpful_test] = use_helpful_feature(cleaned_bigram_test,0);

%Creating the sparse matrices
X_helpful = make_sparse(cleaned_bigram_train_helpful);
X_bigram_helpful = make_bigram_sparse(cleaned_bigram_train_helpful);
Y_helpful = double([cleaned_bigram_train_helpful.rating]');

Xtest_helpful= make_sparse(cleaned_bigram_test_helpful,size(X_helpful,2));
Xtest_bigram_helpful = make_bigram_sparse(cleaned_bigram_test_helpful,size(X_bigram_helpful,2));

%Reducing the number of features
[X_reduced_sw  new_id_sw]= reduce_features(X_sw,0);
[X_bigram new_big_id] = reduce_features(X_bigram,0);
X_combined_helpful = [X_reduced_helpful X_bigram_reduced_helpful];
Xtest_combined_sw = [X_sw_test(:,new_id_sw) Xtest_bigram(:,new_big_id)];

%Training
liblinear_logistic_classifier = liblinear_train(Y,X_combined,'-s 7 -e 1.0 ', 'row');

%Testing
[predicted_label, accuracy, Yhat] = liblinear_predict(ones(size(Xtest_combined,1),1), Xtest_combined, liblinear_logistic_classifier, '-b 0','row');

Yhat = exp(Yhat);
Yhat = bsxfun(@times, Yhat, 1./sum(Yhat,2));
Yhat = sum(bsxfun(@times,Yhat,[1 2 4 5]),2);

%Saving results
save('-ascii', 'submit.txt', 'Yhat');