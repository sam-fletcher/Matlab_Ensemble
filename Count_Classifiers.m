function classifier_counts = Count_Classifiers(params, ensemble)

classifier_counts = zeros(1,length(params.classifier_names)+1);
for c = 1:length(ensemble)
    cls_type = class(ensemble{c});
    if contains(cls_type, "ClassificationECOC") % SVM
        idx = find(params.classifier_names=="SVM",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "ClassificationTree") % DT
        idx = find(params.classifier_names=="DT",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "ClassificationKNN") % kNN
        idx = find(params.classifier_names=="kNN",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "ClassificationDiscriminant") % DA
        idx = find(params.classifier_names=="DA",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "ClassificationNaiveBayes") % NB
        idx = find(params.classifier_names=="NB",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "network") % ANN
        idx = find(params.classifier_names=="ANN",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "ClassificationBaggedEnsemble") % RandomForest
        idx = find(params.classifier_names=="RandomForest",1);
        classifier_counts(idx) = classifier_counts(idx) + 1;
    elseif contains(cls_type, "ClassificationEnsemble")
        if contains(ensemble{c}.Method, "RUSBoost") % RUSBoost
            idx = find(params.classifier_names=="RUSBoost",1);
            classifier_counts(idx) = classifier_counts(idx) + 1;
        elseif contains(ensemble{c}.Method, "AdaBoostM1") % AdaBoost
            idx = find(params.classifier_names=="AdaBoost",1);
            classifier_counts(idx) = classifier_counts(idx) + 1;
        else
            fprintf('\n~~~~ Ensemble ERROR: %i = %s \n', c, ensemble{c}.Method);
        end
    elseif contains(cls_type, "double")
        classifier_counts(end) = classifier_counts(end) + 1;
    else
        fprintf('\n~~ ERROR: %i = %s \n', c, class(ensemble{c}));
    end
end

end
