function [majority_predictions, all_nearest_clusters] = ...
                    Optimize_Majority(all_clusters, all_votes, all_distances, params)
                   
num_classifiers = length(params.classifier_names)*params.att_combinations;
num_records = length(all_votes(:,1));
majority_predictions = zeros(num_records, 1);
all_nearest_clusters = zeros(num_records, 1);

function neg_majority = Biggest_Lead(num_nearest)
    [~, sorted_idxs] = sort(rec_distances);
    nearest_clusters = sorted_idxs(1:ceil(num_nearest));
    nearest_classifiers = zeros(1, length(all_clusters)*num_classifiers);
    for cluster = nearest_clusters
        nearest_classifiers((cluster-1)*num_classifiers+1 : cluster*num_classifiers) = 1;
    end
    nearest_votes = votes(nearest_classifiers==1);
    nearest_votes = nearest_votes(nearest_votes ~= 0);
    maj1 = mode(nearest_votes); 
    maj1_count = sum(nearest_votes==maj1);
    nearest_votes = nearest_votes(nearest_votes ~= maj1);
    if ~isempty(nearest_votes)
        maj2 = mode(nearest_votes);
        maj2_count = sum(nearest_votes==maj2);
        neg_majority = -(maj1_count - maj2_count);
    else
        neg_majority = -maj1_count;
    end
end

fprintf('Optimized Num Nearest Clusters:');
for r = 1:num_records
    votes = all_votes(r,:);
    rec_distances = all_distances(r,:);
    num_nearest_clusters = ceil( fminbnd(@Biggest_Lead, 1, ceil(length(all_clusters)/2)) ); % divide by 2??
    fprintf(' %d', num_nearest_clusters);
    all_nearest_clusters(r) = num_nearest_clusters;
    [~, idx] = sort(rec_distances);
    nrst_clstrs = idx(1:num_nearest_clusters);
    
    nearest_clssfrs = zeros(length(all_clusters)*num_classifiers, 1);
    for clus = nrst_clstrs
        nearest_clssfrs((clus-1)*num_classifiers+1 : clus*num_classifiers) = 1;
    end
    good_votes = votes(nearest_clssfrs==1);
    good_votes = good_votes(good_votes~=0); % remove empty votes (including votes removed by Local GA)
    majority_predictions(r) = mode(good_votes); 
end
fprintf('\n');
end
