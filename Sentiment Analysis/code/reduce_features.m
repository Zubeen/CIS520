function [ reduced_matrix selected_indices] = reduce_features(input,threshold)

selected_indices = find(sum(input) > threshold);
reduced_matrix = input(:,selected_indices);

end

