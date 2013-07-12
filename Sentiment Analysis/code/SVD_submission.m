%% SVD along with liblinear - cross validation

%% Load the data
load ../data/data_no_bigrams.mat;

%% Data cleaning & preprocessing
[cleaned_vocab new_vocab_indices] = clean_vocab(vocab);
[cleaned_train] = clean_data(train,new_vocab_indices);

% Make the training data
X = make_sparse(cleaned_train);
Y = double([cleaned_train.rating]');

% Run svd (takes about an hour to run)
[u s v] = svds(X, 500);
new_X  = X*v;

% Run cross validation
[error rmse] = liblinear_cross_validate(new_X, Y, 5);