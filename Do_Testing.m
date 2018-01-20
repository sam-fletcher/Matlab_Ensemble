
function [majority_error,num_nearest_clusters] = Do_Testing(all_clusters, max_k, ...
                        test_data, test_labels, params)               
% COLLECT PREDICTIONS
% num_pruned = (max_k*(max_k+1) / 2) - length(all_clusters);
% frac_pruned = round( num_pruned / (max_k*(max_k+1) / 2) );
% num_clusters_voting = max_k - frac_pruned
% num_clusters_voting = round( length(all_clusters) / 2 )

num_records = length(test_data(:,1));
num_classifiers = length(params.classifier_names)*params.att_combinations;
majority_predictions = zeros(num_records, 1);
fprintf('\nCollecting Votes on Testing Records for Cluster...\n');

all_records_clusters = cell(length(test_labels), length(all_clusters)); % the clusters that each test record is closest to
all_votes = zeros(num_records, length(all_clusters)*num_classifiers);
curr_vote_idxs = ones(num_records,1);
all_distances = zeros(num_records, length(all_clusters));

% FOR EACH TEST RECORD
for r = 1:num_records
    for c = 1:length(all_clusters)
        rec_dist = norm(test_data(r,:) - all_clusters{c}.centroid); % L2 distance
        all_distances(r,c) = rec_dist;
    end
        
    if params.optimize_majority
        all_records_clusters{r} = [1:length(all_clusters)];
    else
        num_clusters_voting = max_k;
        [~, sorted_idxs] = sort(all_distances(r,:));
        all_records_clusters{r} = sorted_idxs(1:num_clusters_voting); % smallest distances
        
        % RADIUS VERSION. not used.
%         within_radius = zeros(1, length(all_clusters));
%         counter = 1;
%         for c = 1:length(all_clusters)
%             if rec_dist <= all_clusters{c}.radius
%                 within_radius(counter) = c;
%                 counter = counter + 1;
%             end
%         end
%         within_radius = within_radius(within_radius~=0);
%         all_records_clusters{r} = within_radius;
%         frac_within_radius = length(within_radius) / length(all_clusters)

    end
end


for c = 1:length(all_clusters)
    fprintf('%d ', c);
    [all_votes, curr_vote_idxs] = Classify_Records_in_Cluster(c, all_clusters, ...
        all_votes, curr_vote_idxs, all_records_clusters, test_data, params); % includes homog votes
end     

num_nearest_clusters = zeros(num_records, 1);
if params.optimize_majority
    [majority_predictions,num_nearest_clusters] = Optimize_Majority(all_clusters, all_votes, all_distances, params);
else
    for r = 1:num_records
        votes = all_votes(r,:);
        num_nearest_clusters(r) = length(votes)/num_classifiers;
        votes = votes(votes~=0); % remove empty votes (including votes removed by Local GA)
        majority_predictions(r) = mode(votes); 
    end
end

majority_error = mean(majority_predictions ~= test_labels)
fprintf('\n');

end


