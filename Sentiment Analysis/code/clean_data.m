function [Xnew] = clean_data(X,new_indices)

% X is a struct array, new_indices corresponds to index of word in new
% vocab array, Xnew is the struct array with changed word_idx and title_idx
% values

Xnew = X;
for i = 1:numel(X)
    Xnew(i).word_idx = new_indices(X(i).word_idx);
    Xnew(i).title_idx = new_indices(X(i).title_idx);
end


