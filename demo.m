clc;
% clear;
% close all;

%% load data
load ./200Hz_rawdata/dataset.mat;

whos

% X: (1199, 60, 62, 5)
%  1199 : 
%           是样本数量。本数据集包含50个情绪刺激，24名被试。由于丢失1个标签，所以总样本量为24*50-1=1199。
%  60   :
%           是时间窗口的数量。
%  62   :
%           脑电电极的数量。电极顺序见 'chanlocs.mat' 文件。
%  5    :
%           频带数量。频带1-频带5分别为: 
%           $\delta$ (1–3 Hz), $\theta$ (4–7 Hz), $\alpha$ (8–13 Hz), $\beta$ (14–30 Hz), $\gamma$ (31–50 Hz).
%
% Yaro: (50, 60)
%  50   :
%           情绪刺激的数量。
%  60   :
%           情绪刺激的时间（秒），每秒1个情绪强度值。
%
% Yemo: (1199, 1), 每个样本的情绪类型。1,2,3,4,5分别代表sad,angry,fear,disgust,neutral
% Ysub: (1199, 1), 每个样本的被试编号。
% Ytyp: (1199, 1), 暂不公开。
% Yvid: (1199, 1), 每个样本的情绪刺激编号。





%% regression on subject-averaged data

eeg = reshape(X, size(X, 1), size(X, 2), []);

avg_eeg = []; % 40, 60, 310

for i = 1:40
    avg_eeg(i, :, :) = mean(eeg(Yvid == i, :, :), 1);
end

figure

% 设置窗口大小单位为厘米
position = [0, 0, 10, 16]; % [x, y, width, height]，以厘米为单位
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [0 0 40 20]);

predicted_label_list = [];
teY_list = [];

for i = 1:40
    trX = avg_eeg([1:40]~=i, :, :);
    trY = Yaro([1:40]~=i, :);
    
    trX = permute(trX, [3, 1, 2]); % 310, 39 ,60
    trX = reshape(trX, 310, []);
    trX = trX'; % -1, 310
    trY = trY(:);

    teX = avg_eeg([1:40]==i, :, :);
    teY = Yaro([1:40]==i, :);
    
    teX = permute(teX, [3, 1, 2]); % 310, 39 ,60
    teX = reshape(teX, 310, []);
    teX = teX'; % -1, 310
    teY = teY(:);

    model = fitlm(trX,trY);
    predicted_label = predict(model, teX);

    subplot(4, 10, i);
    plot(teY); hold on;
    plot(predicted_label); hold on;
    scatter(0,0, 10, 'filled', 'k'); hold on;
    ylim([0 8]);
    axis('off');
   
    [tempRHO1(i, 1), pp(i, 1) ]= corr(predicted_label(:), teY(:));
    text(10, 6.5, sprintf('rho=%.2f\nlog10(p)=%.2f', tempRHO1(i, 1), log10(pp(i, 1))));

    predicted_label_list = [predicted_label_list; predicted_label];
    teY_list = [teY_list; teY];
end

%% scatter plot

figure
% 计算线性拟合
coefficients = polyfit(teY_list, predicted_label_list, 1);
a = coefficients(1); % 斜率
b = coefficients(2); % 截距

% 生成拟合曲线
x_fit = min(teY_list):0.1:max(teY_list); % 生成拟合曲线上的 x 值
y_fit = polyval(coefficients, x_fit); % 根据拟合系数计算对应的 y 值

% 绘制散点图和拟合曲线
% scatter(x, y, 'filled'); % 绘制散点图
% hold on; % 保持图形，用于绘制曲线
plot(x_fit, y_fit, 'r', 'LineWidth', 2); hold on;% 绘制拟合曲线，'r' 代表红色
scatter(teY_list, predicted_label_list, 'black')
text(3, 0, sprintf('predicted label = %.2f * real label + %.2f', a, b));
axis('off');

%% classification plot

% ch_list = 1:62;


time_avg_X = mean(X(:, :, [1:62], :), 2);

for si = 1:length(unique(Ysub))
    index = find(Ysub == si);
    time_avg_X(index, :) = zscore(time_avg_X(index, :));
end

for si = 1:length(unique(Ysub))
    trainX = time_avg_X(find(Ysub ~= si), :, :, :);
    % trainX = mean(trainX, 2);
    trainX = reshape(trainX, size(trainX,1), []);
    trainY = Yemo(find(Ysub ~= si));

    testX = time_avg_X(find(Ysub == si), :, :, :);
    % testX = mean(testX, 2);
    testX = reshape(testX, size(testX,1), []);
    testY = Yemo(find(Ysub == si));
    
    mdl = fitcecoc(trainX, trainY);
    predict_y = predict(mdl, testX);

    acc_each_sub(si, 1) = mean((predict_y == testY)) * 100;
end

disp(mean(acc_each_sub))
disp(std(acc_each_sub))



