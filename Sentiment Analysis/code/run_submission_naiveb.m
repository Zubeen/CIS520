%% Example submission: Naive Bayes

%% Load the data
load ../data/data_no_bigrams.mat;

% Make the training data
X = make_sparse(train);
Y = double([train.rating]');

categories = unique([train.category]);

% Run training
Yk = bsxfun(@eq, Y, [1 2 4 5]);
nb = nb_train_pk(X'>0, Yk);

for c = 1:size(categories)
      categoryIndices = find([train.category] == categories(c));
      if size(categoryIndices) > 0
          nb_cat(categories(c)) = nb_train_pk(X(categoryIndices,:)',Yk(categoryIndices,:));
      end      
end

%% Make the testing data and run testing
Xtest = make_sparse(test, size(X, 2));
Yhat = zeros(size(Xtest,1),numel(nb.py));

for c = 1:size(categories)
    categoryIndices = find([test.category] == categories(c));
    if size(categoryIndices) > 0
        Yhat(categoryIndices,:) = nb_test_pk(nb_cat(categories(c)),Xtest(categoryIndices,:)');
    end      
end 
%Yhat = nb_test_pk(nb, Xtest'>0);
unkCatIndices = find(sum(Yhat,2) == 0);
Yhat(unkCatIndices,:) = nb_test_pk(nb,Xtest(unkCatIndices,:)'>0);

%% Make predictions on test set
%Convert from classes 1...4 back to the actual ratings of 1, 2, 4, 5
[tmp, Yhat_argmax] = max(Yhat, [], 2);
ratings = [1 2 4 5];
Yhat_argmax = ratings(Yhat_argmax);

Yhat = sum(bsxfun(@times,Yhat,[1 2 4 5]),2);
% Yhat(Yhat > 2 & Yhat < 2.5) = 2;
% Yhat(Yhat >3.5 & Yhat < 4) = 4;
% Yhat(Yhat >= 2.5 & Yhat <= 3.5) = Yhat_argmax(Yhat >= 2.5 & Yhat <= 3.5);

save('-ascii', 'submit.txt', 'Yhat');
