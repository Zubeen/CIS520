%% Stopwords removal along with crossvalidation

%% Load the data
load ../data/data_no_bigrams.mat;

%% Data cleaning & preprocessing
[cleaned_vocab new_vocab_indices] = clean_vocab(vocab);
[cleaned_train] = clean_data(train,new_vocab_indices);

%% Load stop words from text file
stopwords = textread('stopwordlist.txt', '%s');

%% Apply porters stemmer on stopwords as cleaned data is already stemmed
stemmed_stopwords = clean_vocab(stopwords);

%% Find indices of stopwords in vocab
[c ia ib] = intersect(cleaned_vocab, stemmed_stopwords);

%% Remove stopword indices from word_idx and word_count in training examples
for i = 1:size(cleaned_train, 2)
   M = ismember(cleaned_train(i).word_idx, ia);
   word_index = cleaned_train(i).word_idx(M==0);
   wordcount_index = cleaned_train(i).word_count(M==0);
   cleaned_train(i).word_idx = double(word_index);
   cleaned_train(i).word_count = uint8(wordcount_index);
end

% Make the training data
X_stopwords = make_sparse(cleaned_train);
Y = double([cleaned_train.rating]');

% Run cross validation
[error rmse] = liblinear_cross_validate(X_stopwords, Y, 5);

