% X_helpful = X_reduced > 0;
% for i=1:size(X,1)
%     z = X_helpful(i,:);
%     z = z .* helpful_train(i);
%     X_helpful(i,:) = z;
%     i
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [X_reduced  new_id]= reduce_features(X,0);
% [X_bigram_reduced new_big_id] = reduce_features(X_bigram,0);
% X_combined = [X_reduced X_bigram_reduced];
% Xtest_combined = [Xtest(:,new_id) Xtest_bigram(:,new_big_id)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tot_reviews = size(X,1);
% doc_appear = sum(X > 0);
% idf = log(tot_reviews ./ doc_appear);
% 
% X_tfidf = X;
% 
% % num_words = zeros(size(X,1),1);
% % for i=1: size(X,1)
% %     num_words(i) = size(train(i).word_idx,1);
% %     %X_tfidf(i,:) = X_tfidf(i,:)./num_words(i);
% % end
% 
% %X_tfidf = bsxfun(@rdivide,X_tfidf,num_words);
% X_tfidf = bsxfun(@times,X_tfidf,idf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load ../data/data_no_bigrams.mat;

% Make the training data
 %X = make_sparse(cleaned_train);
 %Y = double([cleaned_train.rating]');
% Xtest = make_sparse(test, size(X, 2));

 %nb_cross_validate(X,Y,4);

 %nb = NaiveBayes.fit(X, Y,'Distribution','normal');
 %Yhat_meh  = predict(nb,Xtest,'HandleMissing','on');

%[cleaned_vocab new_indices] = clean_vocab(vocab);
%cleaned_train = clean_data(train,new_indices);