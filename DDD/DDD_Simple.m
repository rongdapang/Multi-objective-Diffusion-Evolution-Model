classdef DDD_Simple < ALGORITHM
% <2026> <multi> <real> <constrained/none>
% DDD-Simple: Simplified Diffusion-based Multi-objective Optimization
%
% This is a simplified version of DDD that works without Deep Learning Toolbox.
% It uses basic Neural Network Toolbox functions for broader compatibility.
%
% noise_schedule --- [0.1, 0.01] --- Noise schedule for diffusion model [start, end]
% network_structure --- [64, 128, 64] --- Hidden layer structure for diffusion network
% ga_generations --- 20 --- Number of GA generations for initial sampling
% sample_size --- 500 --- Size of sample collection for training
% dm_epochs --- 100 --- Number of epochs for training diffusion model
% dm_steps --- 50 --- Number of diffusion steps for sampling
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [noise_schedule, network_structure, ga_generations, sample_size, dm_epochs, dm_steps] = ...
                Algorithm.ParameterSet([0.1, 0.01], [32, 64], 20, 500, 100, 20);

            %% Check for Neural Network Toolbox
            hasNNToolbox = license('test', 'Neural_Network_Toolbox');
            if ~hasNNToolbox
                warning('Neural Network Toolbox not found. Using GA-only mode.');
            end

            %% STAGE 1: High-quality initial sampling with GA
            disp('STAGE 1: Initial sampling with GA...');
            SamplePopulation = InitialSampling(Problem, ga_generations, sample_size);

            %% STAGE 2: Train diffusion model
            DMModel = [];
            if hasNNToolbox
                disp('STAGE 2: Training diffusion model...');
                DMModel = TrainDiffusionModel(Problem, SamplePopulation, noise_schedule, network_structure, dm_epochs);
            end

            %% STAGE 3: Generate initial solutions using diffusion model
            disp('STAGE 3: Generating initial solutions...');
            if ~isempty(DMModel) && DMModel.trained
                Population = GenerateInitialSolutions(Problem, DMModel, dm_steps);
            else
                Population = Problem.Initialization();
            end

            %% STAGE 4: Main optimization loop
            disp('STAGE 4: Main optimization loop...');

            while Algorithm.NotTerminated(Population)
                %% Generate offspring with both GA and DM
                [GAOffspring, DMOffspring] = GenerateOffspring(Problem, Population, DMModel, dm_steps);

                %% Environmental selection using NSGA-II mechanism
                Combined = [Population, GAOffspring, DMOffspring];
                Population = EnvironmentalSelection(Combined, Problem.N);
            end
        end
    end
end

%% ==================== Initial Sampling ====================
function SamplePopulation = InitialSampling(Problem, ga_generations, sample_size)
    % Initialize population
    Population = Problem.Initialization();

    % Run GA
    for gen = 1:ga_generations
        [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
        MatingPool = TournamentSelection(2, length(Population), FrontNo, -CrowdDis);
        Offspring = OperatorGA(Problem, Population(MatingPool));
        Population = EnvironmentalSelection([Population, Offspring], Problem.N);
    end

    % Fill to sample_size with guided mutation if needed
    CurrentSize = length(Population);
    if CurrentSize < sample_size
        ExtraNeeded = sample_size - CurrentSize;
        ExtraDec = zeros(ExtraNeeded, Problem.D);

        for i = 1:ExtraNeeded
            parent = Population(randi(length(Population))).dec;
            mutant = parent + 0.1 * randn(1, Problem.D) .* (Problem.upper - Problem.lower);
            mutant = max(min(mutant, Problem.upper), Problem.lower);
            ExtraDec(i, :) = mutant;
        end

        ExtraPop = Problem.Evaluation(ExtraDec);
        Population = [Population, ExtraPop];
    end

    % Select diverse samples using crowding distance
    if length(Population) > sample_size
        [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
        CrowdDis = CrowdingDistance(Population.objs, FrontNo);
        [~, idx] = sort(-CrowdDis);
        SamplePopulation = Population(idx(1:sample_size));
    else
        SamplePopulation = Population;
    end
end

%% ==================== Diffusion Model Training ====================
function DMModel = TrainDiffusionModel(Problem, Population, noise_schedule, network_structure, epochs)
    % Check for Neural Network Toolbox
    if ~license('test', 'Neural_Network_Toolbox')
        warning('Neural Network Toolbox not found. Using fallback model.');
        DMModel = CreateFallbackModel(Problem);
        return;
    end

    try
        %% 数据准备
        N = length(Population);
        D = Problem.D;
        M = Problem.M;
        
        fprintf('  [DM] Population size: %d, Decision dim: %d, Objective dim: %d\n', N, D, M);

        % 提取决策变量和目标值
        X = reshape([Population.dec], D, N)';  % N x D
        F = reshape([Population.obj], M, N)';  % N x M
        
        fprintf('  [DM] Data extracted: X=[%s], F=[%s]\n', mat2str(size(X)), mat2str(size(F)));

        % 检查数据有效性
        if any(isnan(X(:))) || any(isinf(X(:)))
            error('NaN or Inf detected in decision variables');
        end

        %% 归一化到 [-1, 1]
        X_min = Problem.lower;
        X_max = Problem.upper;
        X_norm = 2 * (X - X_min) ./ (X_max - X_min + 1e-8) - 1;  % N x D
        
        F_min = min(F, [], 1);
        F_max = max(F, [], 1);
        % 避免除零
        F_range = F_max - F_min;
        F_range(F_range < 1e-8) = 1;
        F_norm = (F - F_min) ./ F_range;  % N x M，归一化到 [0,1]
        
        fprintf('  [DM] Normalization: X_norm range [%.3f, %.3f], F_norm range [%.3f, %.3f]\n', ...
            min(X_norm(:)), max(X_norm(:)), min(F_norm(:)), max(F_norm(:)));

        %% 扩散过程参数
        T = 100;
        beta = linspace(noise_schedule(1), noise_schedule(2), T);
        alpha = 1 - beta;
        alpha_bar = cumprod(alpha);
        sqrt_alpha_bar = sqrt(alpha_bar);
        sqrt_one_minus_alpha_bar = sqrt(1 - alpha_bar);
        
        fprintf('  [DM] Diffusion steps T=%d, beta range [%.4f, %.4f]\n', T, beta(1), beta(end));

        %% 准备训练数据（修正维度：行=样本，列=特征）
        samples_per_point = 5;  % 固定为5，平衡速度和质量
        total_samples = N * samples_per_point;
        
        fprintf('  [DM] Generating %d training samples (%d per point)...\n', total_samples, samples_per_point);
        
        % 预分配（行是样本，符合MATLAB习惯）
        train_inputs = zeros(total_samples, D + M + 1);  % [x_t, f_cond, t]
        train_targets = zeros(total_samples, D);         % epsilon (noise)

        % 向量化生成数据
        idx = 1;
        for i = 1:N
            x0 = X_norm(i, :);      % 1 x D
            f_cond = F_norm(i, :);  % 1 x M
            
            % 为每个点生成多个时间步样本
            t_list = randi(T, samples_per_point, 1);
            epsilon = randn(samples_per_point, D);
            
            for j = 1:samples_per_point
                t = t_list(j);
                sqrt_ab = sqrt_alpha_bar(t);
                sqrt_oma = sqrt_one_minus_alpha_bar(t);
                
                % 前向扩散: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
                x_t = sqrt_ab * x0 + sqrt_oma * epsilon(j, :);
                
                train_inputs(idx, 1:D) = x_t;
                train_inputs(idx, D+1:D+M) = f_cond;
                train_inputs(idx, end) = t / T;  % 归一化时间步
                train_targets(idx, :) = epsilon(j, :);
                
                idx = idx + 1;
            end
        end
        
        % 检查训练数据
        if any(isnan(train_inputs(:))) || any(isinf(train_inputs(:)))
            error('NaN or Inf in training inputs');
        end
        if any(isnan(train_targets(:))) || any(isinf(train_targets(:)))
            error('NaN or Inf in training targets');
        end
        
        fprintf('  [DM] Training data ready: inputs=[%s], targets=[%s]\n', ...
            mat2str(size(train_inputs)), mat2str(size(train_targets)));
        fprintf('  [DM] Input range: [%.3f, %.3f], Target range: [%.3f, %.3f]\n', ...
            min(train_inputs(:)), max(train_inputs(:)), min(train_targets(:)), max(train_targets(:)));

        %% 创建和训练神经网络
        fprintf('  [DM] Creating network with structure [%s]...\n', num2str(network_structure));
        
        net = fitnet(network_structure);
        
        % 使用更稳定的训练算法
        net.trainFcn = 'trainscg';  % Scaled Conjugate Gradient，比trainlm更稳定
        
        % 训练参数设置
        net.trainParam.epochs = epochs;
        net.trainParam.showWindow = false;      % 不显示训练窗口
        net.trainParam.showCommandLine = true;  % 显示命令行输出
        net.trainParam.show = 10;               % 每10轮显示一次
        
        % 放宽停止条件，避免早停
        net.trainParam.goal = 1e-3;             % 目标误差（噪声预测不需要极高精度）
        net.trainParam.min_grad = 1e-6;         % 最小梯度
        net.trainParam.max_fail = 10;           % 验证失败10轮才停
        
        % 数据集划分
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.8;       % 80%训练
        net.divideParam.valRatio = 0.2;         % 20%验证
        net.divideParam.testRatio = 0;          % 0%测试

        fprintf('  [DM] Starting training for %d epochs (using %s)...\n', epochs, net.trainFcn);
        
        % 转置为神经网络期望的格式：行=特征，列=样本
        [net, tr] = train(net, train_inputs', train_targets');
        
        fprintf('  [DM] Training completed. Best performance: %.6f, Epochs: %d\n', ...
            tr.best_perf, tr.num_epochs);

        %% 保存模型
        DMModel = struct();
        DMModel.net = net;
        DMModel.X_min = X_min;
        DMModel.X_max = X_max;
        DMModel.F_min = F_min;
        DMModel.F_max = F_max;
        DMModel.T = T;
        DMModel.beta = beta;
        DMModel.alpha = alpha;
        DMModel.alpha_bar = alpha_bar;
        DMModel.sqrt_alpha_bar = sqrt_alpha_bar;
        DMModel.sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar;
        DMModel.D = D;
        DMModel.M = M;
        DMModel.trained = true;

    catch ME
        warning('Diffusion model training failed: %s\nUsing fallback model.', ME.message);
        fprintf('Error details:\n');
        disp(getReport(ME, 'extended'));
        DMModel = CreateFallbackModel(Problem);
    end
end

function DMModel = CreateFallbackModel(Problem)
    DMModel = struct('trained', false, 'D', Problem.D, 'M', Problem.M, ...
                     'X_min', Problem.lower, 'X_max', Problem.upper);
end

%% ==================== Initial Solution Generation ====================
function Population = GenerateInitialSolutions(Problem, DMModel, dm_steps)
    if ~DMModel.trained
        Population = Problem.Initialization();
        return;
    end

    nElite = round(Problem.N * 0.3);
    nDiverse = Problem.N - nElite;

    EliteDec = ConditionalSampling(DMModel, nElite, dm_steps, 'elite');
    DiverseDec = ConditionalSampling(DMModel, nDiverse, dm_steps, 'diverse');

    AllDec = [EliteDec; DiverseDec];
    AllDec = max(min(AllDec, Problem.upper), Problem.lower);
    Population = Problem.Evaluation(AllDec);
end

function X = ConditionalSampling(DMModel, nSamples, nSteps, mode)
    D = DMModel.D;
    M = DMModel.M;
    X = randn(nSamples, D);

    % 设置条件目标向量
    switch mode
        case 'elite'
            F_cond = rand(nSamples, M) * 0.3;  % 精英区域：靠近原点
        otherwise
            F_cond = rand(nSamples, M);        % 多样区域：全范围
    end

    % 归一化条件向量
    F_cond_norm = (F_cond - DMModel.F_min) ./ (DMModel.F_max - DMModel.F_min + 1e-8);
    F_cond_norm = max(0, min(1, F_cond_norm));  % 裁剪到[0,1]
    
    T = DMModel.T;

    % 反向扩散过程
    for step = nSteps:-1:1
        t = max(1, round(step * T / nSteps));
        t_norm = t / T;

        % 组装网络输入
        network_input = [X, F_cond_norm, repmat(t_norm, nSamples, 1)]';
        pred_noise = DMModel.net(network_input)';

        % DDPM采样公式
        alpha_t = DMModel.alpha(t);
        alpha_bar_t = DMModel.alpha_bar(t);
        beta_t = DMModel.beta(t);

        coef1 = 1 / sqrt(alpha_t);
        coef2 = beta_t / (sqrt(1 - alpha_bar_t) + 1e-8);
        sigma_t = sqrt(beta_t);

        % 添加随机噪声（最后一步除外）
        z = randn(nSamples, D) * (t > 1);
        X = coef1 * (X - coef2 * pred_noise) + sigma_t * z;
    end

    % 反归一化到原始决策空间
    X = (X + 1) / 2 .* (DMModel.X_max - DMModel.X_min) + DMModel.X_min;
end

%% ==================== Offspring Generation ====================
function [GAOffspring, DMOffspring] = GenerateOffspring(Problem, Population, DMModel, dm_steps)
    N = Problem.N;
    nDM = round(N * 0.4);   % 40%来自扩散模型
    nGA = N - nDM;          % 60%来自GA

    % GA offspring
    [FrontNo, ~] = NDSort(Population.objs, Population.cons, length(Population));
    CrowdDis = CrowdingDistance(Population.objs, FrontNo);
    MatingPool = TournamentSelection(2, nGA * 2, FrontNo, -CrowdDis);
    GAOffspring = OperatorGA(Problem, Population(MatingPool));
    if length(GAOffspring) > nGA
        GAOffspring = GAOffspring(1:nGA);
    end

    % DM offspring
    if ~isempty(DMModel) && DMModel.trained && nDM > 0
        % 选择多样性好的父代作为条件
        [~, idx] = sort(-CrowdDis);
        selectedIdx = idx(1:min(nDM, length(idx)));
        if length(selectedIdx) < nDM
            selectedIdx = [selectedIdx, randi(length(Population), 1, nDM - length(selectedIdx))];
        end

        F_target = reshape([Population(selectedIdx).obj], DMModel.M, [])';
        DMDec = ArchiveGuidedSampling(DMModel, F_target, dm_steps);
        DMDec = max(min(DMDec, Problem.upper), Problem.lower);
        DMOffspring = Problem.Evaluation(DMDec);
    else
        % 回退到随机采样
        DMDec = Problem.lower + rand(nDM, Problem.D) .* (Problem.upper - Problem.lower);
        DMOffspring = Problem.Evaluation(DMDec);
    end
end

function X = ArchiveGuidedSampling(DMModel, F_target, nSteps)
    nSamples = size(F_target, 1);
    D = DMModel.D;
    X = randn(nSamples, D);

    % 归一化目标向量
    F_norm = (F_target - DMModel.F_min) ./ (DMModel.F_max - DMModel.F_min + 1e-8);
    F_norm = max(0, min(1, F_norm));  % 裁剪到[0,1]
    
    T = DMModel.T;

    % 反向扩散
    for step = nSteps:-1:1
        t = max(1, round(step * T / nSteps));
        t_norm = t / T;

        network_input = [X, F_norm, repmat(t_norm, nSamples, 1)]';
        pred_noise = DMModel.net(network_input)';

        alpha_t = DMModel.alpha(t);
        alpha_bar_t = DMModel.alpha_bar(t);
        beta_t = DMModel.beta(t);

        coef1 = 1 / sqrt(alpha_t);
        coef2 = beta_t / (sqrt(1 - alpha_bar_t) + 1e-8);
        sigma_t = sqrt(beta_t);

        % 减少最后几步的随机性，提高稳定性
        z = randn(nSamples, D) * 0.8 * (t > 1);
        X = coef1 * (X - coef2 * pred_noise) + sigma_t * z;
    end

    X = (X + 1) / 2 .* (DMModel.X_max - DMModel.X_min) + DMModel.X_min;
end

%% ==================== Environmental Selection ====================
function Population = EnvironmentalSelection(Population, N)
    if length(Population) <= N
        return;
    end

    [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
    CrowdDis = CrowdingDistance(Population.objs, FrontNo);

    Selected = FrontNo < MaxFNo;
    LastFront = find(FrontNo == MaxFNo);
    Remain = N - sum(Selected);

    if Remain > 0 && ~isempty(LastFront)
        [~, rank] = sort(-CrowdDis(LastFront));
        Selected(LastFront(rank(1:Remain))) = true;
    end

    Population = Population(Selected);
end