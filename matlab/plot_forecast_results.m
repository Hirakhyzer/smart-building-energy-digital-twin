function plot_forecast_results(results_dir)
%PLOT_FORECAST_RESULTS Read local Python outputs and save a MATLAB figure.
% Usage: plot_forecast_results('outputs')

if nargin < 1
    results_dir = fullfile('..', 'outputs');
end
predictions_path = fullfile(results_dir, 'results', 'forecast_predictions.csv');
if ~isfile(predictions_path)
    error('Missing %s. Run scripts/run_forecast.py first.', predictions_path);
end
T = readtable(predictions_path);
T.timestamp = datetime(T.timestamp);
figure('Color', 'w', 'Position', [100 100 1000 430]);
plot(T.timestamp, T.meter_reading, 'LineWidth', 1.0, 'DisplayName', 'Observed');
hold on;
plot(T.timestamp, T.prediction, 'LineWidth', 1.0, 'DisplayName', 'Predicted');
grid on;
xlabel('Time'); ylabel('Meter reading');
title('Energy forecast evaluation');
legend('Location', 'best');
exportgraphics(gcf, fullfile(results_dir, 'figures', 'forecast_window_matlab.png'), 'Resolution', 250);
end
