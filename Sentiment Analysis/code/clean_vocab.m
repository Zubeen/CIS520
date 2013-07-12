function [cleaned_vocab new_vocab_indices] = clean_vocab(vocab)

% Takes 1xN vocabulary cell array and returns a new 1xM vocabulary cell
% array, along with the indices of the old vocab values in this new vocab
% array.

%Perform all data cleaning in this section, the final step being the
%'unique' operation
vocab = lower(vocab);
for i=1:numel(vocab)
    try
        vocab(i) = {porterStemmer(char(vocab(i)))};
    catch err
        vocab(i) = vocab(i);
    end    
end    

[cleaned_vocab old_order new_vocab_indices] = unique(vocab);
new_vocab_indices = new_vocab_indices';
