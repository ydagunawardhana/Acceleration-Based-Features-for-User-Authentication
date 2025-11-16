function [avg_EER, thresholds, avg_FAR, avg_FRR, eer_idx] = calculateEER(scores, trueLabels, numUsers)
% calculateEER: Calculates the average EER, FAR, and FRR curves

thresholds = 0.01:0.01:0.99;
numThresholds = length(thresholds);

% We will store FAR/FRR for each user, then average
all_FAR = zeros(numUsers, numThresholds);
all_FRR = zeros(numUsers, numThresholds);

for u = 1:numUsers
    % Get Genuine vs. Imposter Scores
    user_u_scores = scores(u, :);
    
    genuine_scores = user_u_scores(trueLabels == u);
    
    imposter_scores = user_u_scores(trueLabels ~= u);
    
    if isempty(genuine_scores)
        warning('No genuine scores found for user %d. Check test labels.', u);
        continue;
    end
    if isempty(imposter_scores)
        warning('No imposter scores found for user %d. Check test labels.', u);
        continue;
    end
    
    % Calculate FAR and FRR at each threshold
    for t_idx = 1:numThresholds
        t = thresholds(t_idx);
        
        % False Acceptance:
        FA = sum(imposter_scores > t);
        all_FAR(u, t_idx) = FA / length(imposter_scores);
        
        % False Rejection:
        FR = sum(genuine_scores < t);
        all_FRR(u, t_idx) = FR / length(genuine_scores);
    end
end

avg_FAR = mean(all_FAR, 1);
avg_FRR = mean(all_FRR, 1);

% Find the Equal Error Rate (EER)
[~, eer_idx] = min(abs(avg_FAR - avg_FRR));

avg_EER = (avg_FAR(eer_idx) + avg_FRR(eer_idx)) / 2;

end