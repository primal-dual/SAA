clear
clc
%n  = 5;                         % dimension of x
%N  = 1000;                      % SAA sample size
%gamma = 1e-2;                   % stepsize
load piecewise_parameter.mat
load x_init.mat
load experiment_results.mat

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
alpha = 1;

p=1;
q = p/(p-1);


est_num_x = 100;
est_num_xi = 100;
for i_n = 1:size(n_values,2)
    n = n_values(i_n);
    D = sqrt(0.5*norm(ones(1,n),p)^2);
    %est_G = 0;
    temp_for_est = randn(est_num_xi,n);
    ls_for_max_x = zeros(1,est_num_x);
    a_row = (1:n)/n;   
    optimal_sol = x_temp_last(i_n, ~isnan(x_init(i_n,:)));
    R = norm(optimal_sol,1);
    for est_i = 1:est_num_x
        x_est = 2*rand(1,n)-1;
        cal_expectation = 0;
        for est_j = 1:est_num_xi
            Xi_current = temp_for_est(est_j,:);
            %inside_value = x_est * (a_row.' + Xi_current.');   % (1xn)*(nx1) -> 1x1
            inside_value = R*(max(0,x_est)-max(0,-x_est)) * (a_row.' + Xi_current.');
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
        gamma = (sqrt(2*alpha)*D)/(est_G*sqrt(N));

        % init per-(n,N) replication storage
        results.rep_eval{i_n, i_N} = nan(1, replications);
        results.rep_time{i_n, i_N} = nan(1, replications);
        results.rep_x{i_n, i_N}    = cell(replications, 1);
        
        obj_over_replications = 0;

        for i_rep = 1:replications
            t_rep = tic;  % start per-rep timer

            Xi = randn(N,n);

            x_0 =R*ones(1,n)/n;
            x_temp = x_0;
            dummy = R - norm(x_temp,1);
            if dummy< -1e-5 
                fprintf('something is wrong')
                return
            end
            y_temp = [max(0,x_temp)/R,max(0,-x_temp)/R,dummy/R];
            gradient = zeros(1,2*n+1);
            obj_val_list = zeros(1,iter);
            gradient_list = zeros(iter,2*n+1);
    
            % track last valid iter for early stop
            i_last = 0;
            k = (1:n)/n;  
            % ===== vectorized inner loop (same structure as second code) =====
            x_temp_list1 = zeros(iter,size(x_temp,2));
            x_temp_list2 = zeros(iter,size(x_temp,2));

            for i = 1:iter
                Xi_current = Xi(i,:);
    
                inside_value = R*(y_temp(1:n)-y_temp(n+1:2*n) ) * (k.' + Xi_current.');   % (1xn)*(nx1) -> 1x1

                [biggest, idxk] = max(v + s * inside_value);    % biggest: scalar, idxk: index
                slope = s(idxk);                                 % scalar
    
                max_vec1 = max(0, R*(y_temp(1:n)-y_temp(n+1:2*n)) - 1);                  % 1 x n
                max_vec2 = max(0, -R*(y_temp(1:n)+y_temp(n+1:2*n)) - 1);                 % 1 x n
                obj = biggest + 0.5 * M * (norm(max_vec1, 2)^2) + 0.5 * M * (norm(max_vec2, 2)^2);
                obj_val_list(i) = obj;
                
                assemble = (k+Xi_current);
                gradient = R*slope * [assemble,-assemble,0];            % 1 x n
                
                part1_1 = R*M*max(0, R*(y_temp(1:n)-y_temp(n+1:2*n)) - 1);
                part1_1(y_temp(1:n) < y_temp(n+1:2*n)+1/R) = 0;
                
                part1_2 = -R*M*max(0, R*(y_temp(1:n)-y_temp(n+1:2*n)) - 1);
                part1_2(y_temp(1:n) < y_temp(n+1:2*n)+1/R) = 0;

                part2_1 = -R*M*max(0, -R*(y_temp(1:n)-y_temp(n+1:2*n)) - 1);
                part2_1(y_temp(1:n) > y_temp(n+1:2*n)-1/R) = 0;
                
                part2_2 = R*M*max(0, -R*(y_temp(1:n)-y_temp(n+1:2*n)) - 1);
                part2_2(y_temp(1:n) > y_temp(n+1:2*n)-1/R) = 0;
                gradient_penalty_part1 = [part1_1,part1_2,0];
                gradient_penalty_part2 = [part2_1,part2_2,0];

                gradient_sum = gradient + gradient_penalty_part1+gradient_penalty_part2;     % 1 x n
    
                
                gradient_list(i,:) = gradient_sum;
    
                numerator   = y_temp .* exp(-gamma*gradient_sum);             % elementwise multiply x_i * e^{-y_i}
                denominator = sum(y_temp .* exp(-gamma*gradient_sum));        % scalar
                y_temp = numerator / denominator;            % elementwise result (n x 1)

                x_temp_list1(i,:) =y_temp(1:n);      
                x_temp_list2(i,:) =y_temp(n+1:2*n);                 
            end
            x_average_temp_1 = sum(x_temp_list1,1)/iter;
            x_average_temp_2 = sum(x_temp_list2,1)/iter;

            x_average = sum(x_average_temp_1-x_average_temp_2,1)/iter;
            Xi_obj   = randn(NN,n);
            t_obj    = R* (k + Xi_obj)*(x_average_temp_1.'-x_average_temp_2.' ) ;      % NN x 1
            W_obj    = t_obj .* s.' + v.';                            % NN x m
            [biggest_vec_obj, idxk_obj] = max(W_obj, [], 2);          % NN x 1
            S_obj    = s(idxk_obj);                                   % NN x 1
    
            max_vec1 = max(0, R*(x_average_temp_1-x_average_temp_2) - 1);                  % 1 x n
            max_vec2 = max(0, -R*(x_average_temp_1+x_average_temp_2) - 1); 
            obj_val_obj = sum(biggest_vec_obj);
            obj_val_obj = obj_val_obj/NN+ 0.5*M*norm(max_vec1,2)^2 + 0.5*M*norm(max_vec2,2)^2;
    
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
outname = sprintf('experiment_results_replications_EMD_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
save(outname, 'results', '-v7.3');
fprintf('Saved results to %s\n', outname);

