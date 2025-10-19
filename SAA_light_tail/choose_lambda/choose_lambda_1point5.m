clear
clc

load piecewise_parameter.mat
load x_init.mat
rng(114514);

% Ensure v and s are column vectors (critical for the vectorized block)
v = v(:);
s = s(:);

lambda_values = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
N = 200;           % SAA sample size
n = 1000;
iter = 800000;       % max iter for subproblem (used as an upper bound)

% --- Early stopping parameters (added) ---
stop_window = 5;    % look at the last 5 objective values
stop_tol    = 1e-6;  % relative tolerance for "no change"
% ----------------------------------------
gamma = 1e-3;

p = 1.5;            % p-norm
m = 10;             % (kept as in your code)

M = 1000;

num_lambda = numel(lambda_values);
obj_val_last = zeros(num_lambda, 1);                 % final obj_val for each n
obj_val = zeros(num_lambda, 1);                      % optimal obj_val for solved solution
x_temp_last  = NaN(num_lambda, n);       % final x_temp row-wise, padded with NaN




for ii = 1:size(lambda_values,2)
    tic
    lambda = lambda_values(ii);
    a_row = (1:n)/n;

    Xi = randn(N,n);
    x_0   = x_init(10, ~isnan(x_init(10,:)));
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

    Xi_obj   = randn(N,n);
    t_obj    = (Xi_obj * x_temp.') + (a_row * x_temp.');  % N x 1
    W_obj    = t_obj .* s.' + v.';                        % N x m
    [biggest_vec_obj, idxk_obj] = max(W_obj, [], 2);      % N x 1
    S_obj    = s(idxk_obj);                               % N x 1
    max_vec1 = max(0, x_temp-1);
    max_vec2 = max(0, -1-x_temp);
    obj_val_obj = sum(biggest_vec_obj);

    obj_val_obj = obj_val_obj/N + 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;

    obj_val(ii) = obj_val_obj;

    fprintf('Finished n=%d | out of sample obj = %.8g (iter_used=%d)\n', n, obj_val(ii), i_last);
    fprintf('lambda =%d\n', lambda);
    toc
end


[~,best_lambda_idx] = min(obj_val);
lambda = lambda_values(best_lambda_idx)
save('../lambda_1point5.mat', 'lambda');
%fprintf('Results saved to experiment_results.mat\n');

