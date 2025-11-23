clc; clear; close all; % Очищення пам'яті та командного вікна
% v.3
% Завантаження даних з файлу CSV
data = readtable("c:\ONEDRIVE\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\!PhD\КОНФЕРЕНЦІЇ\Актуальні проблеми автоматизації та управління Луцький НТУ\FinBench\hmeq.csv");

% Вивід перших 5 рядків для перевірки завантаження
disp('Перші 5 рядків завантажених даних:');
disp(head(data,5));

% Перетворення текстових полів у категоріальні змінні з урахуванням регістру
data.REASON = categorical(data.REASON);
data.JOB = categorical(data.JOB);

% Заповнення пропущених значень (NaN) середніми або медіанними значеннями
data.MORTDUE = fillmissing(data.MORTDUE, 'movmean', 5);
data.VALUE = fillmissing(data.VALUE, 'movmean', 5);
data.YOJ = fillmissing(data.YOJ, 'constant', median(data.YOJ,'omitnan'));
data.DEBTINC = fillmissing(data.DEBTINC, 'constant', median(data.DEBTINC,'omitnan'));

% Визначення цільової змінної
target = 'BAD';

% Відділення ознак від цільової змінної
X = data(:, setdiff(data.Properties.VariableNames, target));
Y = data.(target);

% Розподіл на тренувальну (70%) і тестову (30%) вибірки
cv = cvpartition(height(data), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

% Навчання моделі дерева рішень
fprintf('Навчання моделі дерева рішень...\n');
treeModel = fitctree(XTrain, YTrain);

% Прогнозування на тестовій вибірці
fprintf('Прогнозування результатів на тестовій вибірці...\n');
predLabels = predict(treeModel, XTest);

% Обчислення точності моделі
accuracy = sum(predLabels == YTest) / length(YTest);
fprintf('Точність класифікації﻿ (Accuracy) -\n частка усіх правильно спрогнозованих прикладів\n (і позитивних, і негативних): %.2f%%\n', accuracy * 100);

% Вивід матриці плутанини
confMat = confusionmat(YTest, predLabels);
fprintf('Матриця плутанини:\n');
disp(array2table(confMat, 'VariableNames', {'Прогноз не дефолт', 'Прогноз дефолт'}, 'RowNames', {'Факт не дефолт', 'Факт дефолт'}));

% Обчислення основних метрик класифікації
TP = confMat(2,2);
TN = confMat(1,1);
FP = confMat(1,2);
FN = confMat(2,1);

Recall = TP / (TP + FN);
Precision = TP / (TP + FP);
F1 = 2 * (Precision * Recall) / (Precision + Recall);

fprintf('Повнота (Recall): %.2f%%\n', Recall * 100);
fprintf('Достовірність позитивного прогнозу (Precision) - показує,\nнаскільки модель "влучає" у прогнозі дефолтів\nсеред усіх спрогнозованих дефолтів: %.2f%%\n', Precision * 100);
fprintf('F1-мірa: %.2f%%\n', F1 * 100);

% Побудова ROC-кривої та обчислення площі під кривою (AUC)
fprintf('Побудова ROC-кривої...\n');
[fpRate, tpRate, ~, AUC] = perfcurve(YTest, treeModel.predict(XTest), 1);
fprintf('Площа під кривою (AUC): %.3f\n', AUC);

figure;
plot(fpRate, tpRate, 'LineWidth', 2);
xlabel('Рівень хибних спрацювань (False Positive Rate)');
ylabel('Рівень виявлення (True Positive Rate)');
title('ROC крива моделі дерева рішень');
grid on;
