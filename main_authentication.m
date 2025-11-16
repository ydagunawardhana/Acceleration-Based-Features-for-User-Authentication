% Data Loading and Feature Extraction
% Feature Analysis
% Neural Network Training
% Model Evaluation

% Required these two files extractFeatures.m, calculateEER.m for run main_authentication.m

clc; 
clear; 
close all;

% Parameters
numUsers = 10;       % Based on the 10 users (U1-U10) provided
sampleRate = 30;    
windowSeconds = 2;  
windowSize = sampleRate * windowSeconds;
overlapPercent = 0.5; 
overlap = round(windowSize * overlapPercent);

% Initialization
% Use the 'First Day' (FD) for training and 'Second Day' (MD) for testing.
allFeatures_Train = [];
allLabels_Train = [];
allFeatures_Test = [];
allLabels_Test = [];

% Load Data and Extract Features
disp('Loading data and extracting features...');

for u = 1:numUsers
    fprintf('Processing User %d...\n', u);
    
    % Load Training Data (First Day - FD)
    filename_fd = sprintf('U%dNW_FD.csv', u);
    try
        data_fd = readmatrix(filename_fd);
        
    catch ME
        warning('Could not read %s. Check file path and format.', filename_fd);
        disp(ME.message);
        continue;
    end
    
    % Extract features from FD data
    [features_fd, labels_fd] = extractFeatures(data_fd, u, windowSize, overlap);
    allFeatures_Train = [allFeatures_Train; features_fd];
    allLabels_Train = [allLabels_Train; labels_fd];
    
    % Load Testing Data (Second Day - MD)
    filename_md = sprintf('U%dNW_MD.csv', u);
    try
        data_md = readmatrix(filename_md);
    catch ME
        warning('Could not read %s. Check file path and format.', filename_md);
        disp(ME.message);
        continue;
    end
    
    % Extract features from MD data
    [features_md, labels_md] = extractFeatures(data_md, u, windowSize, overlap);
    allFeatures_Test = [allFeatures_Test; features_md];
    allLabels_Test = [allLabels_Test; labels_md];
end

disp('Feature extraction complete.');
fprintf('Total training samples (windows): %d\n', length(allLabels_Train));
fprintf('Total testing samples (windows): %d\n', length(allLabels_Test));

% Analyze Features
disp('Analyzing feature uniqueness using t-SNE...');

rng('default');
Y = tsne(allFeatures_Train);

figure;
gscatter(Y(:,1), Y(:,2), allLabels_Train);
title('t-SNE (t-distributed Stochastic Neighbor Embedding) Plot of Training Features');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
legend('User 1', 'User 2', 'User 3', 'User 4', 'User 5','User 6','User 7','User 8','User 9','User 10', 'Location', 'best');
grid on;

disp('Feature analysis plot generated.');

% Create the scatter plot
disp('Generating feature-pair scatter plot...');

feature_X_index = 14;  
feature_Y_index = 21;  
feature_X_name = 'X-Axis Mean';
feature_Y_name = 'Y-Axis Mean';

figure;
gscatter(allFeatures_Train(:, feature_X_index), ...
         allFeatures_Train(:, feature_Y_index), ...
         allLabels_Train);
     
title('Scatter Plot (Training Data)');
xlabel(feature_X_name);
ylabel(feature_Y_name);
legend('User 1', 'User 2', 'User 3', 'User 4', 'User 5','User 6','User 7','User 8','User 9','User 10','Location', 'best');
grid on;

disp('Scatter plot generated.');


% Design and Train Neural Network
disp('Training Neural Network Classifier...');

inputs = allFeatures_Train'; 
targets = ind2vec(allLabels_Train');

[inputs_norm, ps] = mapminmax(inputs);

hiddenLayerSize = 25; 
net = patternnet(hiddenLayerSize);

% Train the Network
[net, tr] = train(net, inputs_norm, targets);

disp('Network training complete.');

disp('Evaluating network performance on test data...');

testInputs = allFeatures_Test';
testLabels = allLabels_Test;

testInputs_norm = mapminmax('apply', testInputs, ps);

outputs = net(testInputs_norm);

% Matrix Figure & Model Accurancy
[~, predicted_labels] = max(outputs, [], 1); 
correct_predictions = sum(predicted_labels == testLabels'); 
total_predictions = length(testLabels);
accuracy = correct_predictions / total_predictions;

fprintf('\n Classification Accuracy \n');
fprintf('Accuracy: %.2f %%\n', accuracy * 100);

figure;
cm_nn = confusionchart(testLabels, predicted_labels', ...
    'Title', 'Neural Network Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized'); 

% Calculate EER, FAR, FRR
[avg_EER, thresholds, avg_FAR, avg_FRR, eer_idx] = calculateEER(outputs, testLabels, numUsers);

fprintf('\n Evaluation Results \n');
fprintf('Average Equal Error Rate (EER): %.2f %%\n', avg_EER * 100);

% Plot FAR/FRR Curves
figure;
plot(thresholds, avg_FAR, 'g-', 'LineWidth', 3);
hold on;
plot(thresholds, avg_FRR, 'r-', 'LineWidth', 3);

% Plot the EER point
plot(thresholds(eer_idx), avg_EER, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
text(thresholds(eer_idx) + 0.02, avg_EER, sprintf('EER = %.2f%%', avg_EER*100));

title('Average FAR vs. FRR Curves');
xlabel('Score Threshold');
ylabel('Error Rate');
legend('False Acceptance Rate (FAR)', 'False Rejection Rate (FRR)', 'Equal Error Rate (EER)');
grid on;
hold off;

disp('Evaluation complete.');