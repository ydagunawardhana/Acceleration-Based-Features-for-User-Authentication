function [features, labels] = extractFeatures(data, userID, windowSize, overlap)

% Number of windows
stepSize = windowSize - overlap;
numWindows = floor((size(data, 1) - windowSize) / stepSize) + 1;

numFeatures = 21; 

features = zeros(numWindows, numFeatures);
labels = ones(numWindows, 1) * userID;
window_idx = 1;

for i = 1:stepSize:(size(data, 1) - windowSize + 1)
    
    % Get data (x, y, z columns)
    window_x = data(i : i + windowSize - 1, 1);
    window_y = data(i : i + windowSize - 1, 2);
    window_z = data(i : i + windowSize - 1, 3);
    
    % X-axis time features
    f_x_time(1) = mean(window_x);
    f_x_time(2) = std(window_x);
    f_x_time(3) = rms(window_x);
    f_x_time(4) = min(window_x);
    f_x_time(5) = max(window_x);
    f_x_time(6) = var(window_x);
    
    % Y-axis time features
    f_y_time(1) = mean(window_y);
    f_y_time(2) = std(window_y);
    f_y_time(3) = rms(window_y);
    f_y_time(4) = min(window_y);
    f_y_time(5) = max(window_y);
    f_y_time(6) = var(window_y);

    % Z-axis time features
    f_z_time(1) = mean(window_z);
    f_z_time(2) = std(window_z);
    f_z_time(3) = rms(window_z);
    f_z_time(4) = min(window_z);
    f_z_time(5) = max(window_z);
    f_z_time(6) = var(window_z);
    
    % Calculate feature
    fft_x = abs(fft(window_x));
    f_x_freq(1) = sum(fft_x.^2) / length(window_x); 
    
    fft_y = abs(fft(window_y));
    f_y_freq(1) = sum(fft_y.^2) / length(window_y);
   
    fft_z = abs(fft(window_z));
    f_z_freq(1) = sum(fft_z.^2) / length(window_z);
    
    % Add all features
    features(window_idx, :) = [f_x_time, f_x_freq, f_y_time, f_y_freq, f_z_time, f_z_freq];
    
    window_idx = window_idx + 1;
end

if window_idx <= numWindows
    features(window_idx:end, :) = [];
    labels(window_idx:end) = [];
end

end
