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

iter = 500000;

SAA_penalty = true; % change to false if you dont want any penalty term
if SAA_penalty == false
    lambda = 0;
else
    load lambda_2.mat
end

p = 2; % p-norm

% results container (only keep per-replication data)
results = struct();
results.n_values = n_values;
results.N_values = N_values;

% Only per-replication containers
results.rep_eval = cell(numel(n_values), numel(N_values));  % 1 x replications evals
results.rep_time = cell(numel(n_values), numel(N_values));  % 1 x replications runtimes (sec)
results.rep_x    = cell(numel(n_values), numel(N_values));  % {replications x 1} each is x vector


M = 1000;

NN = 10000; % SAA sample size for out-of-sample evaluation
replications = 5;

for i_n = 1:size(n_values,2)
    for i_N = 1:size(N_values,2)

        % init per-(n,N) replication storage
        results.rep_eval{i_n, i_N} = nan(1, replications);
        results.rep_time{i_n, i_N} = nan(1, replications);
        results.rep_x{i_n, i_N}    = cell(replications, 1);

        obj_over_replications = 0;

        for i_rep = 1:replications
            t_rep = tic;  % start per-rep timer

            n = n_values(i_n);
            N = N_values(i_N);
    
            idx = find(n_values == n, 1);
            if isempty(idx)
                error('n=%d is not in the allowed list.', n);
            end
            gamma = stepsize_values(idx);
    
            a_row = (1:n)/n;                 
    
            Xi = randn(N,n);

            x_0 = x_init(i_n, ~isnan(x_init(i_n,:)));
            x_temp = x_0;
            
            gradient = zeros(1,n);
            obj_val_list = zeros(1,iter);
            gradient_list = zeros(iter,n);
    
            % track last valid iter for early stop
            i_last = 0;
    
            % ===== vectorized inner loop (same structure as second code) =====
            for i = 1:iter
                t = (Xi * x_temp.') + (a_row * x_temp.');       % N x 1
                W = t .* s.' + v.';                              % N x m
    
                [biggest_vec, idxk] = max(W, [], 2);             % N x 1
                S = s(idxk);                                     % N x 1
    
                obj_val_i = sum(biggest_vec);
    
                grad_samples = (sum(S)) * a_row + (S.' * Xi);    % 1 x n
    
                max_vec1 = max(0, x_temp-1);
                max_vec2 = max(0, -1-x_temp);
                obj_val_i = obj_val_i/N + 0.5*lambda*norm(x_temp-x_0,p)^2 ...
                            + 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;
                obj_val_list(i) = obj_val_i;
                
                gradient = grad_samples / N ...
                    + lambda*(norm(x_temp-x_0,p)^(2-p)*sign(x_temp-x_0).*(abs(x_temp-x_0).^(p-1)));
                gradient_penalty1 = M*max(0, x_temp-1);
                gradient_penalty2 = -M*max(0, -1-x_temp);
                gradient_sum = gradient + gradient_penalty1 + gradient_penalty2;
    
                gradient_list(i,:) = gradient_sum;
    
                x_temp_pnorm_power = norm(x_temp,p)^(2-p);
                vec = x_temp_pnorm_power*sign(x_temp).*(abs(x_temp).^(p-1));
                vec(x_temp == 0) = 0;           % avoid NaNs when x_temp == 0
                vec = vec - gamma*gradient_sum;
    
                numerator = sign(vec).*(abs(vec).^(1/(p-1)));
                denominator = norm(vec, p/(p-1))^((2-p)/(p-1));
                optimal_subproblem_solution = numerator/denominator;
            
                x_temp = optimal_subproblem_solution;
                gradient = zeros(1,n);
    
                % early stopping (same logic/names as second code)
                if i >= stop_window
                    recent = obj_val_list(i-stop_window+1:i);
                    base = max(1.0, mean(abs(recent)));
                    max_delta = max(abs(diff(recent)));
                    if (max_delta / base) < stop_tol
                        i_last = i;
                        fprintf('Early stop at iter %d (n=%d, N=%d): recent Δobj=%g (rel %.3g)\n', ...
                                i, n, N, max_delta, max_delta/base);
                        break;
                    end
                end
            end
            if i_last == 0
                i_last = i; % no early stop; finished max iters
            end
            % ==================================================================
    
            % ===== Out-of-sample objective evaluation =====
            tic
            Xi_obj   = randn(NN,n);
            t_obj    = (Xi_obj * x_temp.') + (a_row * x_temp.');      % NN x 1
            W_obj    = t_obj .* s.' + v.';                            % NN x m
            [biggest_vec_obj, idxk_obj] = max(W_obj, [], 2);          % NN x 1
            S_obj    = s(idxk_obj);                                   % NN x 1
    
            max_vec1 = max(0, x_temp-1);
            max_vec2 = max(0, -1-x_temp);
            obj_val_obj = sum(biggest_vec_obj);
            obj_val_obj = obj_val_obj/NN + 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;
    
            fprintf('Eval n=%d, NN=%d | out-of-sample obj = %.8g | rep = %d\n', n, NN, obj_val_obj, i_rep);
            toc

            % record per-rep results
            results.rep_eval{i_n, i_N}(i_rep) = obj_val_obj;   % per-rep OOS evaluation
            results.rep_x{i_n, i_N}{i_rep}    = x_temp;        % per-rep final solution
            results.rep_time{i_n, i_N}(i_rep) = toc(t_rep);    % per-rep total runtime (sec)

            obj_over_replications = obj_over_replications + obj_val_obj;

        % ================================================================================
       end 

        % still compute & print the average for log/debug (not stored)
        obj_over_replications = obj_over_replications/replications;
        fprintf('Average eval for n=%d, N=%d over %d reps = %.8g\n', ...
                n, N, replications, obj_over_replications);

   end
end

% save only per-replication results
outname = sprintf('experiment_results_replications_2_shrink_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
save(outname, 'results', '-v7.3');
fprintf('Saved results to %s\n', outname);
