function [all_votes,curr_vote_idxs] = Classify_Records_in_Cluster(c, all_clusters, ...
                                    all_votes, curr_vote_idxs, all_records_clusters, ...
                                    test_data, params)
                                
classifiers = all_clusters{c}.classifiers;                    
% collect the test records that will use this cluster
close_records = false(length(test_data(:,1)),1);
for r = 1:length(test_data(:,1))
    close_records(r) = ismember(c, all_records_clusters{r});
end
rec_idxs = find(close_records);
cluster_data = test_data(rec_idxs,:);

% FOR EACH CLASSIFIER
for idx = 1:length(classifiers)
    if ~isnumeric(classifiers{idx}) % not homogeneous prediction, or removed
        if contains(class(classifiers{idx}), "network")
            net = classifiers{idx};
            output_weights = net(cluster_data.');
            pred_labels = vec2ind(output_weights);
        else
            try
                pred_labels = predict(classifiers{idx}, cluster_data); % other outputs: [score,node,cnum]
            catch exception
                fprintf('\n %s', exception);
            end
        end
    elseif idx > length(params.classifier_names) && params.limit_homog_votes
        pred_labels(1:length(cluster_data(:,1))) = 0;  
    else % homogeneous prediction (or removed by Local GA, in which case pred=0)
        pred_labels(1:length(cluster_data(:,1))) = classifiers{idx};
    end

    for r = 1:length(rec_idxs) % add new votes to the relevant records
        curr_vote = curr_vote_idxs(rec_idxs(r));
        all_votes(rec_idxs(r),curr_vote) = pred_labels(r);
    end
    curr_vote_idxs(rec_idxs) = curr_vote_idxs(rec_idxs) + 1;
end



end