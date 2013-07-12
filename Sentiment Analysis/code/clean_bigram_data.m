function [Xnew] = clean_bigram_data(X,new_indices)

% X is a struct array, new_indices corresponds to index of word in new
% vocab array, Xnew is the struct array with changed word_idx and title_idx
% values

Xnew = X;
for i = 1:numel(X)
    Xnew(i).bigram_idx = new_indices(X(i).bigram_idx);
end


