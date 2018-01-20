
function [chosen_classifiers,chosen_clusters,diversity,val_error] = ...
            Optimize_Ensemble(all_clusters, validation_data, validation_labels, params)

    function z = Obj_Func(classifier_list)
        % if single: change options & function to @ga. if multi: @gamultiobj
        if ~any(classifier_list) % all zeroes
            z(1) = length(occuring_labels); % z(1)
            z(2) = length(occuring_labels);
% %             z(3) = length(occuring_labels);
            return
        end
            
        current_votes = all_votes(:, classifier_list==1);
        majority_predictions(:,1) = mode(current_votes.');
        if isscalar(majority_predictions) % if there's only one classifier
            majority_predictions = current_votes;
        end
        error = mean(majority_predictions~=validation_labels);
        z(1) = error; % z(1)
% %         SSE = 0;
% %         for lab = 1:length(occuring_labels)
% %             SSE = SSE + mean(majority_predictions(lab_idxs{lab})~=validation_labels(lab_idxs{lab})).^2;
% %         end
% %         z(2) = SSE;
        
        wrong_counts = wrong_pair_counts(classifier_list==1, classifier_list==1);
        double_fault_total = sum(sum(wrong_counts));
        marginal_total = (length(classifier_list)*(length(classifier_list)-1)/2) * length(validation_labels);
        z(2) = double_fault_total / marginal_total; % invariant to number of classifiers. triangle matrix minus diagonal.
    end

    % CONVERT DICTIONARY INTO ARRAY OF CLASSIFIERS
    all_classifiers_inc_homog = [];
    for clust = 1:length(all_clusters)
%         temp = cat( 1, temp, values(all_clusters{clust}.classifiers) );
        all_classifiers_inc_homog = horzcat( all_classifiers_inc_homog, all_clusters{clust}.classifiers );
    end
%     all_classifiers = reshape(temp, 1, []);
    homog_cls = cellfun(@isnumeric, all_classifiers_inc_homog); % also <50% Acc classifiers
    all_classifiers = all_classifiers_inc_homog(~homog_cls); % remove homogeneous predictions - can't use them for Global predictions
    
    if params.with_GA
        all_votes = zeros(length(validation_labels), length(all_classifiers));
        if isunix % LINUX
            parfor c = 1:length(all_classifiers)
                if strcmp(class(all_classifiers{c}), "network")
                    outputs = all_classifiers{c}(validation_data.');
                    converted_outputs = vec2ind(outputs);
                    all_votes(:,c) = converted_outputs.';      
    %             elseif strcmp(class(all_classifiers{c}), "logistic")
    %                 pihat = mnrval(all_classifiers{c}, validation_data);
    %                 converted_outputs = vec2ind(pihat);
    %                 all_votes(:,c) = converted_outputs.';
                else % not neural network or logistic regression
                    all_votes(:,c) = predict(all_classifiers{c}, validation_data);
                end
            end
            options = optimoptions(@gamultiobj, 'PopulationType','bitstring', 'PopulationSize',params.pop_size);
        else % NOT LINUX
            for c = 1:length(all_classifiers)
                if strcmp(class(all_classifiers{c}), "network")
                    outputs = all_classifiers{c}(validation_data.');
                    converted_outputs = vec2ind(outputs);
                    all_votes(:,c) = converted_outputs.';
    %             elseif strcmp(class(all_classifiers{c}), "logistic")
    %                 pihat = mnrval(all_classifiers{c}, validation_data);
    %                 converted_outputs = vec2ind(pihat);
    %                 all_votes(:,c) = converted_outputs.';
                else % not neural network or logistic regression
                    all_votes(:,c) = predict(all_classifiers{c}, validation_data);
                end
            end
%             options = optimoptions(@ga, 'PopulationType','bitstring', 'PlotFcn',@gaplotbestf);
            options = optimoptions(@gamultiobj, 'PopulationType','bitstring', 'PlotFcn',@gaplotpareto,'PopulationSize',params.pop_size);
        end

%         [optimal_ens, err, exitflag, output] = ... % if uncommented: change options function x2 above
%             ga(@Obj_Func, length(all_classifiers), [],[],[],[],[],[],[], options);
        wrong_votes = zeros(size(all_votes));
        wrong_votes(:,:) = all_votes(:,:) ~= validation_labels(:); 
        wrong_pair_counts = zeros( length(wrong_votes(1,:)),length(wrong_votes(1,:)) );
        for i = 1:length(wrong_votes(1,:)) % each column (classifier)
            for j = i+1:length(wrong_votes(1,:)) % triangle matrix
                wrong_pair_counts(i,j) = sum(wrong_votes(:,i) & wrong_votes(:,j));
            end
        end
        occuring_labels = unique(validation_labels);
        lab_idxs = cell(1, length(occuring_labels));
        for i = 1:length(occuring_labels)
            lab_idxs{i} = validation_labels == occuring_labels(i);
        end            
        [pareto_ens,fval] = gamultiobj(@Obj_Func, length(all_classifiers), [],[],[],[],[],[], options);
        
        lowest_errors = pareto_ens(fval(:,1)==min(fval(:,1)), : ); % use if z(1) = error
        % Use the following if z(1) is NOT error:
%         errors = ones(length(pareto_ens(:,1)), 1);
%         for i = 1:length(pareto_ens(:,1)) % calculate prediction errors
%             curr_votes = all_votes(:, pareto_ens(i,:)==1);
%             if size(curr_votes,2) > 1
%                 maj_preds(:,1) = mode(curr_votes.');
%             else
%                 maj_preds(:,1) = curr_votes;
%             end
%             errors(i) = mean(maj_preds~=validation_labels);
%         end
%         lowest_errors = pareto_ens(errors==min(errors), : ); % filter by lowest error

        ensemble_sizes = sum(lowest_errors.');
        optimal_ens = lowest_errors(ensemble_sizes==min(ensemble_sizes), : ); % filter by smallest size
        objective_values = fval(ismember(pareto_ens,optimal_ens(1,:),'rows'), : )
        diversity = objective_values(1,end);
        val_error = objective_values(1,1);
        chosen_classifiers = all_classifiers(optimal_ens(1,:) == 1);

        original_indexes = find(homog_cls==0);
        chosen_classifier_indexes = original_indexes(optimal_ens(1,:) == 1);
        chosen_clusters = ceil( chosen_classifier_indexes ./ (length(params.classifier_names)*params.att_combinations) );
    
    else % skip Global GA
        if params.proximity_voting % was params.with_local_GA
            chosen_classifiers = all_classifiers_inc_homog;
        else
            chosen_classifiers = all_classifiers;
        end
        diversity = -1;
        chosen_clusters = ceil( find(homog_cls==0) ./ (length(params.classifier_names)*params.att_combinations) );
    end
end
