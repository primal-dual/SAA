%generating initial solution
clear
clc
rng(114514);                 
n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 5000];
num_n = numel(n_values);
x_init  = NaN(num_n, max(n_values));       % final x_temp row-wise, padded with NaN

for i = 1:length(n_values)
    n = n_values(i);
    x_generate = rand(1,n)-0.5;
    x_init(i,1:n) = x_generate;
end
save('x_init.mat','x_init')