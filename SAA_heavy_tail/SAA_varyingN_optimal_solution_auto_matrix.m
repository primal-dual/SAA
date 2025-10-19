clear
clc

load piecewise_parameter.mat
load x_init.mat
rng(114514);

% Ensure v and s are column vectors (critical for the vectorized block)
v = v(:);
s = s(:);

n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 5000];
N = 10000;           % SAA sample size
iter = 500000;       % max iter for subproblem (used as an upper bound)

% --- Early stopping parameters (added) ---
stop_window = 10;    % look at the last 10 objective values
stop_tol    = 1e-6;  % relative tolerance for "no change"
% ----------------------------------------

stepsize_values = [
    5e-4;  % n = 100
    8e-4;  % n = 200
    8e-4;  % n = 300
    8e-4;  % n = 400
    1e-3;  % n = 500
    1e-3;  % n = 600
    1e-3;  % n = 700
    1e-3;  % n = 800
    1e-3;  % n = 900
    1e-3;  % n = 1000
    1e-3;  % n = 1500
    1e-3;  % n = 2000
    1e-3   % n = 5000
];

SAA_penalty = false; % change to false if you dont want any penalty term
if SAA_penalty == false
    lambda = 0;
else
    error('lambda needs to be chosen!')
end
p = 1.5;            % p-norm
m = 10;             % (kept as in your code)
highest = 3.01;     % Power-law distribution with a=highest -1 and here a is mean
xi_min  = 1;        % Power-law distribution with b=xi_min
M = 100000;

num_n = numel(n_values);
obj_val_last = zeros(num_n, 1);                 % final obj_val for each n
obj_val = zeros(num_n, 1);                      % optimal obj_val for solved solution
x_temp_last  = NaN(num_n, max(n_values));       % final x_temp row-wise, padded with NaN


for ii = 1:num_n
    n = n_values(ii);

    idx = find(n_values == n, 1);
    if isempty(idx)
        error('n=%d is not in the allowed list.', n);
    end
    gamma = stepsize_values(idx);

    a_row = (1:n)/n;

    % Generate Xi with zero-mean (Pareto-shifted)
    temp = rand(N, n);
    expectation = (highest-1)/(highest-2);
    Xi = xi_min * (1 + (temp.^(-1/(highest-1)) - 1)) - expectation;

    empirical_mean = mean(Xi(:));
    fprintf('n=%d | empirical mean (should be ~0): %g | expectation: %g\n', n, empirical_mean, expectation);

    %x_0   = x_init(ii, ~isnan(x_init(ii,:)));
    x_0 = zeros(1,n);
    x_temp = x_0;

    gradient = zeros(1, n);
    obj_val_list = zeros(1, iter);
    i_last = 0; % (added) to record the last valid iteration index

    for i = 1:iter

        t = (Xi * x_temp.') + (a_row * x_temp.');      % N x 1
        W = t .* s.' + v.';                             % N x m

        [biggest_vec, idxk] = max(W, [], 2);            % N x 1
        S = s(idxk);                                    % N x 1 (selected slope per sample)

        obj_val_i = sum(biggest_vec);

        % = (sum(S))*a_row + S.' * Xi
        grad_samples = (sum(S)) * a_row + (S.' * Xi);   % 1 x n

        max_vec1 = max(0, x_temp-1);
        max_vec2 = max(0, -1-x_temp);
        obj_val_i = obj_val_i/N + 0.5*lambda*norm(x_temp,p)^2 ...
                    + 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;
        obj_val_list(i) = obj_val_i;

        gradient = grad_samples / N ...
                 + lambda*(norm(x_temp,p)^(2-p)*sign(x_temp).*(abs(x_temp).^(p-1)));
        gradient_penalty1 = M*max(0, x_temp-1);
        gradient_penalty2 = -M*max(0, -1-x_temp);
        gradient_sum = gradient + gradient_penalty1 + gradient_penalty2;

        x_temp_pnorm_power = norm(x_temp,p)^(2-p);
        vec = x_temp_pnorm_power * sign(x_temp).*(abs(x_temp).^(p-1));
        vec(x_temp == 0) = 0;
        vec = vec - gamma * gradient_sum;

        numerator   = sign(vec) .* (abs(vec).^(1/(p-1)));
        denominator = norm(vec, p/(p-1))^((2-p)/(p-1));
        optimal_subproblem_solution = numerator / denominator;

        x_temp = optimal_subproblem_solution;
        gradient = zeros(1, n);   

        if i >= stop_window
            recent = obj_val_list(i-stop_window+1:i);
            base = max(1.0, mean(abs(recent)));  % avoid tiny denominators
            max_delta = max(abs(diff(recent)));
            if (max_delta / base) < stop_tol
                i_last = i;
                fprintf('Early stop at iter %d (n=%d): recent Δobj=%g (rel %.3g)\n', ...
                        i, n, max_delta, max_delta/base);
                break;
            end
        end
    end
    if i_last == 0
        i_last = i; % no early stop; finished max iters
    end

    obj_val_last(ii)    = obj_val_list(i_last);
    x_temp_last(ii,1:n) = x_temp;

    temp_obj = rand(N, n);
    Xi_obj   = xi_min * (1 + (temp_obj.^(-1/(highest-1)) - 1)) - expectation;
    t_obj    = (Xi_obj * x_temp.') + (a_row * x_temp.');  % N x 1
    W_obj    = t_obj .* s.' + v.';                        % N x m
    [biggest_vec_obj, idxk_obj] = max(W_obj, [], 2);      % N x 1
    S_obj    = s(idxk_obj);                               % N x 1
    max_vec1 = max(0, x_temp-1);
    max_vec2 = max(0, -1-x_temp);
    obj_val_obj = sum(biggest_vec_obj);

    obj_val_obj = obj_val_obj/N + 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;

    obj_val(ii) = obj_val_obj;

    fprintf('Finished n=%d | final obj = %.8g (iter_used=%d)\n', n, obj_val_last(ii), i_last);
end

save('experiment_results.mat', 'n_values', 'obj_val_last', 'x_temp_last');
fprintf('Results saved to experiment_results.mat\n');

