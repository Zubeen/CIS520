tot_reviews = size(X,1);
doc_appear = sum(X > 0);
idf = log(tot_reviews ./ doc_appear);
idf = idf(new_id);

X_tfidf = X_reduced;


num_words = zeros(size(X,1),1);
for i=1: size(X,1)
    num_words(i) = double(size(cleaned_bigram_train(i).word_idx,1));
    %X_tfidf(i,:) = X_tfidf(i,:)./num_words(i);
end

X_tfidf = bsxfun(@rdivide,X_tfidf,num_words);
X_tfidf = bsxfun(@times,X_tfidf,idf);