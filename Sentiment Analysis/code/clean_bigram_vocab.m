function [cleaned_bigram_vocab new_vocab_indices] = clean_bigram_vocab(vocab)

% Takes 1xN vocabulary cell array and returns a new 1xM vocabulary cell
% array, along with the indices of the old vocab values in this new vocab
% array.

%Perform all data cleaning in this section, the final step being the
%'unique' operation
t = CTimeleft(numel(vocab));
for i=1:numel(vocab)
    t.timeleft(); 
    try
        split_str = regexp(char(vocab(i)),'_','split');
        split_str(1) = {porterStemmer(char(split_str(1)))};
        split_str(2) = {porterStemmer(char(split_str(2)))};
        vocab(i) = strcat(split_str(1),'_',split_str(2));
    catch err
        vocab(i) = vocab(i);
    end    
end    

[cleaned_bigram_vocab old_order new_vocab_indices] = unique(vocab);
new_vocab_indices = new_vocab_indices';
