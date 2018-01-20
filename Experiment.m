clear
experiment_note = "# Multi-Obj, 100 Bags, 0.5 Val (0.5 Train), 10 folds, 5 tests, 9 Classifiers, Pop Size 200, 2 MultiObj, one-hot-enc, normalized"
% ICONIP = no filter, proximity voting, no GA, K=3root, 6 classifiers
total_time = tic;

parent_path = cd(cd('..'));
params.path = [parent_path, filesep, 'Datasets', filesep];
params.datasets = {
      'Sonar' 'Heart' 'Ionosphere' 
%     'Hepatitis'   'Glass'  'SPECT'   'Haberman'  ... % < 500
%     'Iris' 'Adult' ...
%     'ClimateModel' 'Blood' 'Balance' 'WBC' 'AusCredit' 'PimaDiabetes' 'Mammographic' 'Vehicle' ... % < 1000
%     'Biodeg' 'Banknotes'  'Segmentation' ... % < 5000
%     'Waveform1' 'Waveform2' 'PageBlocks' 'Landsat' ...
%     'Letters' ... % < 20000 % LETTERS UPDATED TO NUM_TESTS=1
%     'Nursery'
    % 'Yeast' 'Ozone' 'Wine-Red'
        }; % 
%  'Bupa' 'Spect-F', 'GammaTele', 'WhiteWine', ,  'PenDigits' 
%%% Big Attribute Counts: Biodeg, Ozone, Waveform2
%%% Do 10-fold instead, and remove Yeast?
params.num_folds = 10;
params.num_tests = 5;

params.root_max_k = 100; % nthroot(N) = max_k, unless not kmeans, then = num clusterings.
params.validation_frac = 0.5; % ignored if both GAs are false. if set to -1, OOB error is used instead.
params.pop_size = 200; % default = 200
params.bag_size = 1.0; % if 0.5, N*0.5 samples used, etc. # if -1, uses all records
params.classifier_names = ["SVM","DT","kNN","DA","NB","RUSBoost","ANN","RandomForest","AdaBoost"]; % 
params.feature_subsets = 1; % *num_classifiers ## if 1, use all features ## if -1, dynamically decide
params.AB_fraction = 1; % fraction of attributes to use per bag. does NOT change the number of classifiers made. requires feature_subsets=1

params.with_kmeans = false;
params.with_pruning = false; % Jaccard Pruning. probably turn off if not using kmeans
params.clust_simil_frac = 0.9; % threshold fraction of records that are similar
params.trivial_filter = 20; % *num_labels ## -1 = num_attributes

params.proximity_voting = false;
params.with_GA = true; % Global
params.with_local_GA = false;
params.with_local_pruning = false; % 50% Acc Pruning per cluster.
params.optimize_majority = false; % max(maj1-maj2) from closest half of clusters.
params.limit_homog_votes = false; % homog clusters only get num_types_classifiers votes
params.training_error = false;
params.with_PCA = false;

warning off stats:classreg:learning:modifier:AdaBoostM2:modify:NonPositiveLoss;
warning off stats:classreg:learning:modifier:AdaBoostM1:modify:Terminate;
warning off stats:mnrfit:IterOrEvalLimit;

if isunix
    mypool = parpool(16);
end

time = char(datetime('now','Format','d-MMM__HH-mm-ss'))
results_file = [parent_path, filesep, 'Results', filesep, time, '.txt'];
fileID = fopen(results_file, 'a');

param_string = sprintf('# Training_Error=%i \tCluster_Simil_Threshold=%.2f \tk=%.2froot \tClassifiers: %s', ...
                    params.training_error, params.clust_simil_frac, params.root_max_k, sprintf('%s ',params.classifier_names));
header = ["Dataset", ...
    "Error_Maj_Vote", "SD_Maj_Vote", ...
    "Num_Clusters", "Num_Clusters_SD", "Homog_Clusters", "Clust_Maj_Fraction", ...
    "Clust_Maj_Frac_Excl_Homog", ...
    "Diversity", "Val_Error", "Converge_Time", "Fold_Time", ...
    "Num_Nearest_Voting_Clusters", "Num_Nearest_Voting_Clusters_SD", ...
    "All_Homog_Positions", "All_Pruned_Positions", "All_Kept_Positions", "All_Trivial_Positions", ...
    "Classifer_Counts_preGA", "Classifer_Counts_postGA", "All_Classifier_Positions", ...
    "All_Errors"
%     "Classify_Runtime", "Classify_Runtime_SD", "Network_Runtime", "Network_Runtime_SD", "Jaccard_Runtime", "Jaccard_Runtime_SD"
    ];
fprintf(fileID, '\n\n%s\n', time);
fprintf(fileID, '%s\n', experiment_note);
fprintf(fileID, '%s\n', param_string);
fprintf(fileID, '%s\t', header);
fprintf(fileID, '\n');
fclose(fileID);

            
for d = 1:length(params.datasets)
    if strcmp(params.datasets{d}, 'Letters')
        params.num_tests = 1; % to speed up experiments
    end
    d_start = tic;
    dat_file = string([params.path, char(params.datasets{d}), filesep, 'data.txt'])
    att_file = string([params.path, char(params.datasets{d}), filesep, 'attributes.txt']); 
    attributes = importdata(att_file);
    label_idx = find(attributes(1,:) == 2) % class label index
    categ_idxs = find(attributes(1,:) == 0) % categorical indexes
%     data = importdata(dat_file);
    table = readtable(dat_file, 'ReadVariableNames', false);
%     data_properties = data.Properties
    labels = grp2idx( table2array(table(:,label_idx)) ); % convert categ labels to integers
    unique_labels = unique(labels)


    fprintf('%s: Example record, before encoding: \n', params.datasets{d});% sprintf('%s ', table{1,:}));
    head(table,1)
    % ONE-HOT ENCODING:
    new_columns = [];
    for cat_idx = 1:width(table(1,:))
        if attributes(1,cat_idx) == 0
            cat_column = grp2idx( table2array(table(:,cat_idx)) ); % convert to integers
            encoded = ind2vec(cat_column.').'; % convert to binary encoding
            new_columns = [new_columns encoded]; % concatenate
        end
    end
    % now remove the original categ columns (+ class), once we don't need the indices
    table(:,[categ_idxs label_idx]) = [];
%     table(:, label_idx) = [];
    data = table2array(table);
    data = [data new_columns];
    data = full(data); % make sure it's not sparse (can happen from encoding)
    data( :, ~any(data,1) ) = []; % remove empty columns (dimension=1)
    rmmissing(data); % remove rows with NaNs
    num_features = length(data(1,:));
    params.att_combinations = params.feature_subsets; % pre-defined subset count, sampled randomly with replacement
    if params.feature_subsets < 0 % dynamic subset count, sampled randomly with replacement
        params.att_combinations = ceil( nchoosek(num_features, ceil(num_features/2)) / length(params.classifier_names));
        if num_features > 7
            params.att_combinations = ceil(35 * log10( nchoosek(num_features, ceil(num_features/2)) ) / length(params.classifier_names));
        end
    end
    fprintf('%s: Example record (without class): \n%s\n', params.datasets{d}, sprintf('%d ', full(data(1,:))));
    % NORMALIZATION
    for i = 1:length(data(1,:))
        data(:,i) = mapminmax(data(:,i).').';
    end
    fprintf('%s: Example record after normalization: \n%s\n', params.datasets{d}, sprintf('%d ', full(data(1,:))));

    num_repeats = params.num_tests*params.num_folds;
    maj_errors = [];
%     runtime_stats = cell(1, (params.num_tests*params.num_folds));
    num_clusters = [];
    chosen_clusters = cell(1, num_repeats); % cluster IDs that each chosen classifier belongs to (including homog)
    homog_positions = cell(1, num_repeats);
    pruned_positions = cell(1, num_repeats);
    kept_positions = cell(1, num_repeats);
    trivial_positions = cell(1, num_repeats);
    classifier_counts_pre_GA = cell(1, num_repeats);
    classifier_counts_post_GA = cell(1, num_repeats);
    nearest_voting_clusters = cell(1, num_repeats);
    av_maj_ratios = [];
    diversity = [];
    val_error = [];
    converge_time = [];
    fold_time = [];
    for t = 0:params.num_tests-1
        CV_object = cvpartition(labels, 'KFold', params.num_folds);
        % FOR EACH FOLD
        for f = 1:params.num_folds
            fld_tme = tic;
            i = t*params.num_folds + f;
            fprintf('\n%s: Test+Fold: %i\n', params.datasets{d}, i);
            [ maj_errors(i), num_clusters(i), chosen_clusters{i}, ...
                homog_positions{i}, pruned_positions{i}, kept_positions{i}, trivial_positions{i}, ...
                classifier_counts_pre_GA{i}, classifier_counts_post_GA{i}, nearest_voting_clusters{i}, ...
                av_maj_ratios(i), diversity(i), val_error(i), converge_time(i) ] = Do_Fold(f, CV_object, params, data, labels, unique_labels);
            fold_time(i) = toc(fld_tme)
        end
    end
    
    num_homog = []*num_repeats;
    for i = 1:num_repeats
        num_homog(i) = length(homog_positions{i});
    end
    
    clsfy_counts_pre = zeros(1,length(params.classifier_names)+1); % +1 for locally removed classifiers
    clsfy_counts_post = zeros(1,length(params.classifier_names)+1);
    for c = 1:length(params.classifier_names)+1
        for i = 1:num_repeats
            clsfy_counts_pre(c) = clsfy_counts_pre(c) + classifier_counts_pre_GA{i}(c);
            clsfy_counts_post(c) = clsfy_counts_post(c) + classifier_counts_post_GA{i}(c);
        end
    end        
    
    maj_error_av = mean(maj_errors)
    maj_error_std = std(maj_errors)
    clsfy_string_pre = sprintf('%.2f ', clsfy_counts_pre./num_repeats);
    clsfy_string_post = sprintf('%.2f ', clsfy_counts_post./num_repeats);
    all_errors = sprintf('%.4f ', maj_errors);
    all_homog = [];
    all_pruned = [];
    all_kept = [];
    all_trivial = [];
    all_clsfy_clstring_homes = [];
    all_nearest_voting_clusters = [];
    for i = 1:num_repeats
%         non_homog = kept_positions{i}; % the clusterings that the non-homog kept clusters belong to
%         for h = 1:length(homog_positions{i})
%             non_homog(find(non_homog==homog_positions{i}(h),1)) = [];
%         end
        clustering_homes = zeros(1,length(chosen_clusters{i}));
        for c = 1:length(chosen_clusters{i})
            clustering_homes(c) = kept_positions{i}(chosen_clusters{i}(c));
        end
%         all_homog = strcat(all_homog, sprintf('%i ', homog_positions{i}), '_');
        all_clsfy_clstring_homes = horzcat(all_clsfy_clstring_homes, clustering_homes);
        all_homog = horzcat(all_homog, homog_positions{i});
        all_pruned = horzcat(all_pruned, pruned_positions{i});
        all_kept = horzcat(all_kept, kept_positions{i});
        all_trivial = horzcat(all_trivial, trivial_positions{i});    
        all_nearest_voting_clusters = horzcat(all_nearest_voting_clusters, nearest_voting_clusters{i}.');   
    end
    
    if ~isempty(all_homog)
        [counts,~] = histcounts(all_homog, max(all_homog)-min(all_homog)+1);
        homog_string = sprintf('%ix%.2f ', [ [min(all_homog):max(all_homog)]; counts./num_repeats ]);
    else
        homog_string = "None ";
    end
    if ~isempty(all_pruned)
        [counts,~] = histcounts(all_pruned, max(all_pruned)-min(all_pruned)+1);
        pruned_string = sprintf('%ix%.2f ', [ [min(all_pruned):max(all_pruned)]; counts./num_repeats ]);
    else
        pruned_string = "None ";
    end
    if ~isempty(all_trivial)
        [counts,~] = histcounts(all_trivial, max(all_trivial)-min(all_trivial)+1);
        trivial_string = sprintf('%ix%.2f ', [ [min(all_trivial):max(all_trivial)]; counts./num_repeats ]);    
    else
        trivial_string = "None ";
    end
    if ~isempty(all_kept)
        [counts,~] = histcounts(all_kept, max(all_kept)-min(all_kept)+1);
        kept_string = sprintf('%ix%.2f ', [ [min(all_kept):max(all_kept)]; counts./num_repeats ]);
    else
        kept_string = "None ";
    end      
    if ~isempty(all_clsfy_clstring_homes)
        [counts,~] = histcounts(all_clsfy_clstring_homes, max(all_clsfy_clstring_homes)-min(all_clsfy_clstring_homes)+1);
        clsfy_homes_string = sprintf('%ix%.2f ', [ [min(all_clsfy_clstring_homes):max(all_clsfy_clstring_homes)]; counts./num_repeats ]);  
    else
        clsfy_homes_string = "None ";
    end 
    
    fileID = fopen(results_file, 'a');
    formatString = ['%s ', ...
            '\t%.4f \t%.4f ', ...
            '\t%.2f \t%.2f \t%.2f \t%.4f ', ...
            '\t%.4f ', ...
            '\t%.5f \t%.5f \t%.2f \t%.2f ', ...
            '\t%.2f \t%.2f ', ...
            '\t%s \t%s \t%s \t%s ', ...
            '\t%s \t%s \t%s ', ...
            '\t%s'
            ];
    fprintf(fileID, formatString, params.datasets{d}, ...
        mean(maj_errors), std(maj_errors), ...
        mean(num_clusters), std(num_clusters), mean(num_homog), mean(av_maj_ratios), ...
        mean(av_maj_ratios) - mean(num_homog)/mean(num_clusters), ... % av maj excl. homog clusters
        mean(diversity), mean(val_error), mean(converge_time), mean(fold_time), ...
        mean(all_nearest_voting_clusters), std(all_nearest_voting_clusters), ...
        homog_string, pruned_string, kept_string, trivial_string, ...
        clsfy_string_pre, clsfy_string_post, clsfy_homes_string, ...
        all_errors );
    fprintf(fileID, '\n');
    fclose(fileID);
    toc(d_start)
end % datasets

toc(total_time)
if isunix
    delete(mypool);
end
% if we don't need graphics or want to save memory, execute: matlab -nojvm
