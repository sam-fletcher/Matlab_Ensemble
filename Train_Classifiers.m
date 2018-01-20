%{
- Start with all default classifier settings. Can improve later.
- Add in Random Feature Subspaces.
%}
function classifiers = Train_Classifiers(params, cluster_data, cluster_labels, glob_val_data, glob_val_labels)

function error = Obj_Func(classifier_list)
    current_votes = all_votes(:, classifier_list==1);
    majority_predictions(:,1) = mode(current_votes.'); 
    error = mean(majority_predictions ~= val_labels);
end

% training_times = containers.Map;

if ~params.with_local_GA
    train_data = cluster_data;
    train_labels = cluster_labels;
    val_data = glob_val_data;
    val_labels = glob_val_labels;
else % with local GA
%     rand_idxs = randperm(length(cluster_labels), ceil(length(cluster_labels)*params.validation_frac));
%     rest = setdiff( [1:length(cluster_labels)], rand_idxs );
    cv = cvpartition(cluster_labels, 'HoldOut', params.validation_frac);
    val_data = cluster_data(cv.test(),:); % or test(cv) ?
    val_labels = cluster_labels(cv.test(),:);
    train_data = cluster_data(cv.training(),:);
    train_labels = cluster_labels(cv.training(),:);
end

occuring_labels = unique(train_labels);
num_features = length(cluster_data(1,:));
classifiers = cell(1, length(params.classifier_names) * params.att_combinations);
feature_perms = [];
if params.feature_subsets < 0 && num_features < 22 % for memory reasons
    feature_perms = nchoosek(1:num_features, ceil(num_features/2));
end

if length(occuring_labels) == 1 % homogeneous
    for idx = 1:length(classifiers)
        classifiers{idx} = double(occuring_labels(1));
    end
else % heterogeneous
    for idx = 1:length(classifiers)
        if params.feature_subsets == 1
%             rand_features = []; % the atts to NOT use 
            rand_features = randperm(num_features, floor(num_features*(1-params.AB_fraction)));
        else
            if ~isempty(feature_perms)
                p_idx = randi(length(feature_perms));
                chosen = feature_perms(p_idx);
                feature_perms(p_idx) = [];
                rand_features = setdiff(1:num_features, chosen); % the atts to NOT use 
            else
                rand_features = randperm(num_features, floor(num_features/2)); % w/o replacement
            end
        end
        train_subset = train_data;
        train_subset(:,rand_features) = 0;
        name = params.classifier_names( mod(idx-1,length(params.classifier_names))+1 );
        
        if strcmp(name, "DT")
%             args = [];
            classifiers{idx} = fitctree(train_subset, train_labels);%, 'Name',datasample(args,1) );
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "RUSBoost")
            classifiers{idx} = fitcensemble(train_subset, train_labels, 'Method','RUSBoost');
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "AdaBoost")
            if length(occuring_labels) == 2
                classifiers{idx} = fitcensemble(train_subset, train_labels, 'Method','AdaBoostM1');
            else % multiclass
                classifiers{idx} = fitcensemble(train_subset, train_labels, 'Method','AdaBoostM2');
            end
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "RandomForest")
            classifiers{idx} = fitcensemble(train_subset, train_labels, 'Method','Bag');
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "SVM")
%             args = [[1, 10, 100] % BoxConstraints
%                     [true, false] % ClipAlphas
%                     ['linear', 'polynomial', 'rbf'] % KernelFunction
%                     [0, 0.01, 0.05] % OutlierFraction
%                     ['ISDA', 'L1QP'] % Solver
%                     
%                     [] % PolynomialOrder, if KernelFunction=polynomial ?
%                     ];
%             if length(occuring_labels) == 2
%                 classifiers{idx} = fitcsvm(train_subset,train_labels, 'KernelFunction',datasample(args,1), ...
%                                 'Standardize',true, 'KernelScale','auto');
%             else %multi-class
%                 template = templateSVM('KernelFunction',datasample(args,1), ...
%                                 'Standardize',true, 'KernelScale','auto');
                classifiers{idx} = fitcecoc(train_subset, train_labels);%, 'Learners',template); 
%             end
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "ANN")
            net = patternnet(10); % 'trainlm'
            net.trainParam.goal = 0.00001; % default = 0
            net.divideParam.trainRatio = 80/100;
            net.divideParam.valRatio = 20/100;
            net.trainParam.showWindow = false;
            targets = dummyvar(train_labels);
            [classifiers{idx},~] = train(net, train_subset.', targets.');
%         elseif strcmp(name, "LR")
%             classifiers(char(name)) = mnrfit(cluster_data, cluster_labels);
            % the fitglm function is for only 2 labels
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "kNN") % either numer or categ, not both
            classifiers{idx} = fitcknn(train_subset, train_labels);
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "DA") % can't handle categorical
            try
                classifiers{idx} = fitcdiscr(train_subset, train_labels, 'DiscrimType','diaglinear');
            catch exception
                fprintf("\n~~~ DA failed because: %s\n", exception.identifier);
                classifiers{idx} = 0;
            end
        % ------------------------------------------------------------------------------------------------------
        elseif strcmp(name, "NB")
            classifiers{idx} = fitcnb(train_subset, train_labels, 'Distribution','kernel');
        else
            fprintf("\n~~~ ERRROR: %s is not known\n", name);
        end
    end % training classifiers
    

    if params.with_local_GA || params.with_local_pruning
        all_votes = zeros(length(val_labels), length(classifiers));
        for clsfy = 1:length(classifiers)
            if ~isnumeric(classifiers{clsfy})
                if strcmp(class(classifiers{clsfy}), "network")
                    outputs = classifiers{clsfy}(val_data.');
                    converted_outputs = vec2ind(outputs);
                    all_votes(:,clsfy) = converted_outputs.';
                else % not neural network
                    all_votes(:,clsfy) = predict(classifiers{clsfy}, val_data);
                end
                
                if params.with_local_pruning
                    % REMOVE CLASSIFIERS WITH LESS THAN or equal 50% ACCURACY ON average LABEL
                    errors = zeros(1,length(occuring_labels));
                    for lab = 1:length(occuring_labels)
                        recs = val_labels == occuring_labels(lab);
                        errors(lab) = mean(all_votes(recs,clsfy) ~= val_labels(recs));
                    end
                    if mean(errors) > 0.5
                        classifiers{clsfy} = 0;
                    end
                end
            end
        end
        if params.with_local_GA
            % GENETIC OPTIMIZATION
            options = optimoptions(@ga, 'PopulationType','bitstring', 'Display','off');
            optimal_ens = ga(@Obj_Func, length(classifiers), [],[],[],[],[],[],[], options);
            for clsfy = 1:length(classifiers)
                if optimal_ens(clsfy)==0
                    classifiers{clsfy} = 0;
                end
            end
        end
    end % Local GA or Local 50% Pruning
    
end % heterogeneous
    
end

