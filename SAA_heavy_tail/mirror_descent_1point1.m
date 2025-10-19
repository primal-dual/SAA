clear
clc
%n  = 5;                         % dimension of x
%N  = 1000;                      % SAA sample size
%gamma = 1e-2;                   % stepsize
load piecewise_parameter.mat
load x_init.mat

rng(114514);                

% ensure column vectors (match second code)
v = v(:);
s = s(:);

n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 5000];
N_values = [200, 400, 600];

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

% early stopping params (same names as second code)
stop_window = 10;
stop_tol    = 1e-6;

%iter = 500000;

SAA_penalty = false; % change to false if you dont want any penalty term
if SAA_penalty == false
    lambda = 0;
else
    load lambda_1point5.mat
end

p = 1.1; % p-norm

% results container (only keep per-replication data)
results = struct();
results.n_values = n_values;
results.N_values = N_values;

% Only per-replication containers
results.rep_eval = cell(numel(n_values), numel(N_values));  % 1 x replications evals
results.rep_time = cell(numel(n_values), numel(N_values));  % 1 x replications runtimes (sec)
results.rep_x    = cell(numel(n_values), numel(N_values));  % {replications x 1} each is x vector

% pareto & penalty params
highest = 3.01;
expectation = (highest-1)/(highest-2);
xi_min = 1;
M = 1000;

NN = 10000; % SAA sample size for out-of-sample evaluation
replications = 5;
alpha = 1;

q = p/(p-1);


est_num_x = 100;
est_num_xi = 100;
for i_n = 1:size(n_values,2)
    n = n_values(i_n);
    D = sqrt(0.5*norm(ones(1,n),p)^2);
    %est_G = 0;
    temp_for_est = rand(est_num_xi,n);
    ls_for_max_x = zeros(1,est_num_x);
    a_row = (1:n)/n;   
    for est_i = 1:est_num_x
        x_est = 2*rand(1,n)-1;
        cal_expectation = 0;
        for est_j = 1:est_num_xi
            Xi_current = xi_min * (1 + (temp_for_est(est_j,:).^(-1/(highest-1)) - 1)) - expectation;
            inside_value = x_est * (a_row.' + Xi_current.');   % (1xn)*(nx1) -> 1x1
            [biggest, idxk] = max(v + s * inside_value);    % biggest: scalar, idxk: index
            slope = s(idxk);                                 % scalar
            gradient = slope * (a_row + Xi_current);            % 1 x n
            gradient_norm_square = norm(gradient,q)^2;
            cal_expectation = cal_expectation + gradient_norm_square;
        end
        cal_expectation = cal_expectation/est_num_xi;
        ls_for_max_x(est_i) = cal_expectation;
    end
    est_G = max(ls_for_max_x);
    for i_N = 1:size(N_values,2)
        N = N_values(i_N);
        iter = N;
        gamma = (sqrt(2*alpha)*D)/(est_G*sqrt(N))*0.1;

        % init per-(n,N) replication storage
        results.rep_eval{i_n, i_N} = nan(1, replications);
        results.rep_time{i_n, i_N} = nan(1, replications);
        results.rep_x{i_n, i_N}    = cell(replications, 1);

        obj_over_replications = 0;

        for i_rep = 1:replications
            t_rep = tic;  % start per-rep timer
    
            % Generate Xi with zero-mean (Pareto-shifted)
            temp = rand(N,n);
            Xi = xi_min*(1+(temp.^(-1/(highest-1)) -1)) - expectation;
            
            empirical_mean = mean(Xi(:));
            fprintf('n=%d, N=%d | empirical mean (should be ~0): %g | expectation: %g\n', ...
                     n, N, empirical_mean, expectation);
            
            %x_0 = x_init(i_n, ~isnan(x_init(i_n,:)));
            x_0 = zeros(1,n);
            x_temp = x_0;
            
            gradient = zeros(1,n);
            obj_val_list = zeros(1,iter);
            gradient_list = zeros(iter,n);
    
            % track last valid iter for early stop
            i_last = 0;
            k = (1:n)/n;  
            % ===== vectorized inner loop (same structure as second code) =====
            x_temp_list = zeros(iter,size(x_temp,2));
            for i = 1:iter
                Xi_current = xi_min * (1 + (temp(i,:).^(-1/(highest-1)) - 1)) - expectation;
    
                inside_value = x_temp * (k.' + Xi_current.');   % (1xn)*(nx1) -> 1x1

                [biggest, idxk] = max(v + s * inside_value);    % biggest: scalar, idxk: index
                slope = s(idxk);                                 % scalar
    
                max_vec1 = max(0, x_temp - 1);                  % 1 x n
                max_vec2 = max(0, -1 - x_temp);                 % 1 x n
                obj = biggest + 0.5 * M * (norm(max_vec1, 2)^2) + 0.5 * M * (norm(max_vec2, 2)^2);
                obj_val_list(i) = obj;
    
                gradient = slope * (k + Xi_current);            % 1 x n

                gradient_penalty = M * max_vec1 - M * max_vec2; % 1 x n
                gradient_sum = gradient + gradient_penalty;     % 1 x n
    
                
                gradient_list(i,:) = gradient_sum;
    
                x_temp_pnorm_power = norm(x_temp,p)^(2-p);
                vec = x_temp_pnorm_power*sign(x_temp).*(abs(x_temp).^(p-1));
                vec(x_temp == 0) = 0;           % avoid NaNs when x_temp == 0
                vec = vec - gamma*gradient_sum;
    
                numerator = sign(vec).*(abs(vec).^(1/(p-1)));
                denominator = norm(vec, p/(p-1))^((2-p)/(p-1));
                optimal_subproblem_solution = numerator/denominator;
            
                x_temp = optimal_subproblem_solution;
                x_temp_list(i,:) =x_temp; 
            end
            x_average = sum(x_temp_list,1)/iter;
            temp_obj = rand(NN, n);
            Xi_obj   = xi_min * (1 + (temp_obj.^(-1/(highest-1)) - 1)) - expectation;
            t_obj    = (Xi_obj * x_average.') + (a_row * x_average.');      % NN x 1
            W_obj    = t_obj .* s.' + v.';                            % NN x m
            [biggest_vec_obj, idxk_obj] = max(W_obj, [], 2);          % NN x 1
            S_obj    = s(idxk_obj);                                   % NN x 1
    
            max_vec1 = max(0, x_average-1);
            max_vec2 = max(0, -1-x_average);
            obj_val_obj = sum(biggest_vec_obj);
            obj_val_obj = obj_val_obj/NN + 0.5*lambda*norm(x_average,p)^2 ...
                          + 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;
    
            fprintf('Eval n=%d, NN=%d | out-of-sample obj = %.8g | rep = %d\n', n, NN, obj_val_obj, i_rep);

            % record per-rep results
            results.rep_eval{i_n, i_N}(i_rep) = obj_val_obj;   % per-rep OOS evaluation
            results.rep_x{i_n, i_N}{i_rep}    = x_average;        % per-rep final solution
            results.rep_time{i_n, i_N}(i_rep) = toc(t_rep);    % per-rep total runtime (sec)

            obj_over_replications = obj_over_replications + obj_val_obj;

        end
                % still compute & print the average for log/debug (not stored)
        obj_over_replications = obj_over_replications/replications;
        fprintf('Average eval for n=%d, N=%d over %d reps = %.8g\n', ...
                n, N, replications, obj_over_replications);
    end
end

% save only per-replication results
outname = sprintf('experiment_results_replications_SMD_1point1_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
save(outname, 'results', '-v7.3');
fprintf('Saved results to %s\n', outname);

