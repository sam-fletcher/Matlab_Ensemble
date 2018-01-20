
function [ maj_err, num_clusters, chosen_clusters, ...
    homog_positions, pruned_positions, kept_positions, trivial_positions, ...
    classifier_counts_pre_GA, classifier_counts_post_GA, nearest_voting_clusters, ...
    av_maj_ratio, diversity, val_error, converge_time] = Do_Fold(f, CV_object, params, data, labels, unique_labels)

if ~params.training_error
    test_data = data(CV_object.test(f),:); % testing data
    test_labels = labels(CV_object.test(f),:);
else % training error
    test_data = data(CV_object.training(f),:);
    test_labels = labels(CV_object.training(f),:);
end
 
if params.with_GA && ~params.with_local_GA % GLOBAL GA
    train_val_data = data(CV_object.training(f),:); % training data
    train_val_labels = labels(CV_object.training(f),:);
    
    if params.with_PCA % FEATURE REDUCTION WITH PCA
        [coeffs,score,~,~,explained,mu] = pca(train_val_data);
        explanatory_power_of_princ_components = explained
        best_components = find(cumsum(explained) < 99.99);
        train_val_data = score(:, best_components);
        centered = bsxfun(@minus, test_data, mu); 
        test_data = centered*coeffs(:, best_components);
    end
    
    if params.validation_frac < 0 % USE OOB ERROR INSTEAD
        val_data = []; % defined later, during Bagging
        val_labels = [];
        train_data = train_val_data;
        train_labels = train_val_labels;
    else
        cv = cvpartition(train_val_labels, 'HoldOut', params.validation_frac);
        val_data = train_val_data(cv.test(),:);
        val_labels = train_val_labels(cv.test(),:);
        train_data = train_val_data(cv.training(),:);
        train_labels = train_val_labels(cv.training(),:); 
    end
    
else % with local GA, or no GA at all
    val_data = [];
    val_labels = [];
    train_data = data(CV_object.training(f),:);
    train_labels = labels(CV_object.training(f),:);
end


if ~params.with_kmeans
    max_k = params.root_max_k; % max_k = total number of bags
elseif params.root_max_k >= 1
    max_k = ceil( nthroot(length(train_labels), params.root_max_k) ); % max number of clusterings = cuberoot(N)
else
    max_k = ceil( length(train_labels) * params.root_max_k );
end
max_clusters = sum(1:max_k)
all_clusters = cell(1, max_clusters);
% runtime_stats = cell(1, max_clusters);
homog_positions = zeros(1, max_clusters);
pruned_positions = zeros(1, max_clusters);
kept_positions = zeros(1, max_clusters);
trivial_positions = zeros(1, max_clusters);
maj_ratios = zeros(1, max_clusters);

counter = 0; % +1 for each cluster created and stored
for k = 1:max_k % the k-th clustering
    if params.with_kmeans
        % K-MEANS CLUSTERING
%         with_label = cat(2, train_data, train_labels);
        if isunix
            [cluster_idx, centroids, sumD, allD] = kmeans(train_data, k, 'MaxIter', 500, 'Replicates',5,...
                                                            'Options',statset('UseParallel',1) );
        else
            [cluster_idx, centroids, sumD, allD] = kmeans(train_data, k, 'MaxIter', 500, 'Replicates',5 );
        end
%     else
%         % RANDOM disjoint SUBSETS
%         cluster_idx = randi(k, length(train_data), 1);
%         centro = cell(k, length(train_data(1,:) ));
%         for i = 1:k
%             rec_idxs = cluster_idx == i;
%             centro{i} = mean(train_data(rec_idxs,:)); 
%         end
%         centroids = cell2mat(centro);
    end
    

    for i = 1:k % the i-th cluster in the k-th clustering. IF NOT CLUSTERING, STOPS AFTER i=1
        new = true;
        pos = sum(1:k)-1 + i;
        if ~params.with_kmeans % BAGGING
            if params.bag_size < 0
                rec_idxs = 1:length(train_labels);
            else
                rec_idxs = randsample(length(train_labels), ceil(length(train_labels)*params.bag_size), true); % n or n/2? with replacement
            end
            centroid = mean(train_data(rec_idxs,:));
            occuring_labels = unique(train_labels(rec_idxs));
            if params.validation_frac < 0      
                val_idxs = setdiff(1:length(train_labels), rec_idxs); % out-of-bag records
                val_data = train_data(val_idxs, :); % define validation data
                val_labels = train_labels(val_idxs);
            end
        else % K-MEANS
            rec_idxs = find(cluster_idx == i);
            centroid = centroids(i,:);
            av_radius = sumD(i) / length(rec_idxs);
            max_radius = max( allD(rec_idxs) );
        
            i_size = length(rec_idxs);
            occuring_labels = unique(train_labels(rec_idxs));    

            if i_size < params.trivial_filter*length(occuring_labels) || ...
                    (params.trivial_filter<0 && i_size<length(train_data(1,:)))
                new = false;
                trivial_positions(pos) = k;
            else              
                % PRUNING
                tic; 
                if params.with_pruning
                    for j = 1:counter
                        j_size = length(all_clusters{j}.records);
                        % clusters can only be similar if they're close in size.
                        % checking this first is at least 8x faster.
                        if i_size*params.clust_simil_frac <= j_size && ...
                                j_size*params.clust_simil_frac <= i_size
                            % Jaccard 
                            jac = length(intersect(rec_idxs, all_clusters{j}.records)) / ...
                                length(union(rec_idxs, all_clusters{j}.records));
                            if jac >= params.clust_simil_frac % if very similar
                                new = false;
                                pruned_positions(pos) = k;
                                break
                            end
                        end
                    end
                end
            end % end of k-means stuff
        end
        if new==true
            counter = counter + 1;
            all_clusters{counter}.records = rec_idxs;
            all_clusters{counter}.centroid = centroid;
            if params.with_kmeans
                all_clusters{counter}.radius = max_radius;
                all_clusters{counter}.av_dist = av_radius;
            end
%             runtime_stats{counter}.jaccard = toc;
            kept_positions(pos) = k;
            
            [~,freq] = mode(train_labels(rec_idxs));
            maj_ratios(counter) = max(freq) / length(rec_idxs);
            % Homogeneous Clusters        
            if length(occuring_labels) == 1 % homogeneous
                homog_positions(pos) = k; % store the clustering number
            end    
        end  
        if ~params.with_kmeans
            break;
        end
    end
end
all_clusters(cellfun('isempty', all_clusters)) = []; % remove empty cells
homog_positions = homog_positions(homog_positions ~= 0);
pruned_positions = pruned_positions(pruned_positions ~= 0);
kept_positions = kept_positions(kept_positions ~= 0);
trivial_positions = trivial_positions(trivial_positions ~= 0);
maj_ratios = maj_ratios(maj_ratios ~= 0);
av_maj_ratio = mean(maj_ratios);
num_clusters = length(all_clusters);
fprintf('Actual Number of Clusters = %d\n', num_clusters);
fprintf('Num Trivially Skipped = %d\n', length(trivial_positions));
fprintf('Num Pruned = %d\n', length(pruned_positions));
fprintf('Num Kept = %d\n', length(kept_positions));
fprintf('Num Homog = %d\n', length(homog_positions));

% TRAIN CLASSIFIERS IN EACH CLUSTER
fprintf('Training %d Classifiers for Clusters... ', length(params.classifier_names)*params.att_combinations);
if isunix % do parallel
    parfor c = 1:length(all_clusters)
%         fprintf('%d ', c);
        cluster_data = train_data(all_clusters{c}.records,:); % cluster data
        cluster_labels = train_labels(all_clusters{c}.records); % labels for the records in the cluster
        all_clusters{c}.classifiers = ...
            Train_Classifiers(params, cluster_data, cluster_labels, val_data, val_labels);
    end
else % do single thread
    for c = 1:length(all_clusters)
        fprintf('%d ', c);
        cluster_data = train_data(all_clusters{c}.records,:); % cluster data
        cluster_labels = train_labels(all_clusters{c}.records); % labels for the records in the cluster
%         runtime_stats{c}.N = length(cluster_data);
        all_clusters{c}.classifiers = ...
            Train_Classifiers(params, cluster_data, cluster_labels, val_data, val_labels);
    end
end

% we'll just count the homogeneous clusters
% for c = 1:length(all_clusters)    
%     for clsfer = 1:length(params.classifier_names)
%         if isnumeric(all_clusters{c}.classifiers(char(params.classifier_names(clsfer)))) % homogeneous prediction
%             
%         end
%     end
% end
all_classifiers_inc_homog = [];
for clust = 1:length(all_clusters)
    all_classifiers_inc_homog = horzcat( all_classifiers_inc_homog, all_clusters{clust}.classifiers );
end
classifier_counts_pre_GA = Count_Classifiers(params, all_classifiers_inc_homog)
if sum(classifier_counts_pre_GA(1:end-1))==0
    maj_err = 1;
    diversity = 1;
    val_error = 1;
    chosen_clusters = [];
    nearest_voting_clusters = [];
    classifier_counts_post_GA = classifier_counts_pre_GA;
    return;
end
% GENETIC GLOBAL OPTIMIZATION
% This function will output all classifiers if global GA is turned off.
fprintf('\n');
cnvg = tic;
[optimal_ensemble, chosen_clusters, diversity, val_error] = Optimize_Ensemble(all_clusters, val_data, val_labels, params);
converge_time = toc(cnvg);
classifier_counts_post_GA = Count_Classifiers(params, optimal_ensemble)


% DO TESTING
if params.proximity_voting
    % USE LOCAL PREDICTIONS - includes homog clusters
    [maj_err,nearest_voting_clusters] = Do_Testing(all_clusters, max_k, test_data, test_labels, params); 
else
    % if not proximity_voting, optimal_ensemble doesn't include homog votes
    majority_predictions = zeros(length(test_labels), 1);
    nearest_voting_clusters = zeros(length(test_labels), 1);
    all_votes = zeros(length(test_labels), length(optimal_ensemble));
    for c = 1:length(optimal_ensemble)
        if strcmp(class(optimal_ensemble{c}), "network")
            outputs = optimal_ensemble{c}(test_data.');
            converted_outputs = vec2ind(outputs);
            all_votes(:,c) = converted_outputs.';
    %     elseif strcmp(class(optimal_ensemble{c}), "logistic")
    %         pihat = mnrval(optimal_ensemble{c}, test_data);
    %         all_votes(:,c) = vec2ind(pihat);
        else % not neural network or logistic regression
            all_votes(:,c) = predict(optimal_ensemble{c}, test_data);
        end
    end
    for r = 1:length(test_labels)
        votes = all_votes(r,:);
        nearest_voting_clusters(r) = length(votes) / (length(params.classifier_names)*params.att_combinations);
        votes = votes(votes~=0); % remove empty votes
        majority_predictions(r) = mode(votes); 
    end
    maj_err = mean(majority_predictions ~= test_labels);
end


end
