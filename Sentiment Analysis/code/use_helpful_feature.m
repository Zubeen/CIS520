
function [ new_train helpful ] = use_helpful_feature(train, c)
%Takes as input a structure array and assigns weights to each review 
% based on the helpful feature. c is a penalization term.

helpful = zeros(numel(train),1);
for i=1:numel(train)
    try
        if size(train(i).helpful,2) > 0
            split_str = regexp(train(i).helpful,'of','split');
            w = str2double(char(split_str(1))) / str2double(char(split_str(2)));
            helpful(i) = 1+ w + c;
        else
            helpful(i) = 1 - c;
        end   
    catch err
    end    
end

t = CTimeleft(numel(train));
new_train = train;
for i = 1:numel(train)
    t.timeleft();  
    new_train(i).word_count = double(new_train(i).word_count) * helpful(i);
end

end

