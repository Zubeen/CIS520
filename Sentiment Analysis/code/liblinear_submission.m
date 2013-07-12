%% Example submission: Naive Bayes

%% Load the data
% load ../data/data_no_bigrams.mat;

% Make the training data
% X = make_sparse(train);
% Y = double([train.rating]');

% Run training
% Yk = bsxfun(@eq, Y, [1 2 4 5]);
liblinear_logistic_classifier = liblinear_train(Y,double(X_combined > 0),'-s 7 -e 1.0', 'row');
% nb = nb_train_pk([X]', [Yk]);

%% Make the testing data and run testing
% Xtest = make_sparse(test, size(X, 2));
[predicted_label, accuracy, Yhat] = liblinear_predict(ones(size(Xtest_combined,1),1), double(Xtest_combined > 0) , liblinear_logistic_classifier, '-b 0','row');

Yhat = exp(Yhat);
Yhat = bsxfun(@times, Yhat, 1./sum(Yhat,2));

%% Make predictions on test set

% Convert from classes 1...4 back to the actual ratings of 1, 2, 4, 5
% [tmp, Yhat] = max(Yhat, [], 2);
% ratings = [1 2 4 5];
% Yhat = ratings(Yhat)';
Yhat = sum(bsxfun(@times,Yhat,[1 2 4 5]),2);

Yhat (find(Yhat > 2 & Yhat < 2.75))=2;
Yhat (find(Yhat > 3.25 & Yhat < 4))=4;

save('-ascii', 'submit.txt', 'Yhat');
