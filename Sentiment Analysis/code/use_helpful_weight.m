function [Xnew] = use_helpful_weight(X,helpful)

% X is a struct array, helpful is a COL vector with N entries, where N is
% the number of entries in X. Returns a new struct array.

t = CTimeleft(numel(X));
Xnew = X;
for i = 1:numel(X)
    t.timeleft();  
    Xnew(i).word_count = double(Xnew(i).word_count) * helpful(i);
end
