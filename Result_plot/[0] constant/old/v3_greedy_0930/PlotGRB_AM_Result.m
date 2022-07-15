%% Plot AM and Gurobi result 
close all
graph_size = 20;
num_samples = length(Gurobi_costlog);
Gurobi_costlog = -round(Gurobi_costlog);
AM_costlog = -round(AM_costlog);
diff = round(Gurobi_costlog-AM_costlog);
temp = [sum(diff==0); sum(diff==1); sum(diff==2); sum(diff>2)];
% Set marker size
% Normalize temp
temp = (temp-min(temp)) / (max(temp)-min(temp));
temp_factor = temp*100 + 10;
sz = zeros(num_samples,1);
for i=1:num_samples
    if diff(i) > 3 
        diff(i) = 3; 
    end
    sz(i) = temp_factor(diff(i)+1);
end
%% Begin plot
figure(1)
plot(min(Gurobi_costlog)-1:graph_size,min(Gurobi_costlog)-1:graph_size, 'linewidth', 2); grid on; hold on;
scatter(Gurobi_costlog, AM_costlog, sz, 'filled')
title("Gurobi Result vs Attention Model Result (N=" + graph_size + ")")
xlabel('GRB'); ylabel('AM');
legend('Gurobi Result','Attention Model Result')
rho = corrcoef(Gurobi_costlog, AM_costlog) % correlation coeff;
[max_v, argmax] = max(Gurobi_timelog)
saveas(gcf,'[Plot1] Gurobi vs AM.png')

figure(2)
boxplot(Gurobi_timelog, 'DataLim', [0, 100], 'ExtremeMode', 'compress'); grid on;
title("Gurobi Computation Time (N=" + graph_size + ")")
ylabel('Time (sec)')
saveas(gcf,'[Plot2] Gurobi Computation Time.png')

figure(3)
plot(1:num_samples, Gurobi_timelog); grid on; 
xlabel("Sample Count"); ylabel("Time")
saveas(gcf,'[Plot1] Gurobi Time.png')

% Compute Percentage Error
pdiff = abs(diff)./(Gurobi_costlog) * 100;

sum(pdiff<5)