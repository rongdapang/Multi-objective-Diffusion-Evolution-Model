classdef SolutionArchive < handle
% SolutionArchive - Manages elite solutions for diffusion model training
%
% This class implements an archive for storing and managing elite solutions
% discovered during optimization. It supports:
%   - Non-dominated sorting for admission control
%   - Hypervolume-based contribution calculation
%   - Age-weighted replacement strategy
%   - Stratified sampling for training data
%
% Properties:
%   Capacity - Maximum number of solutions in archive
%   M - Number of objectives
%   Solutions - Array of SOLUTION objects
%   Ages - Age of each solution (generations since addition)
%   HypervolumeContributions - Hypervolume contribution of each solution
%   Size - Current number of solutions in archive
%
% Methods:
%   add(solutions) - Add new solutions to archive
%   getTrainingData(n) - Get n samples for training
%   getReferencePoints(n) - Generate n reference points for conditioning

    properties (SetAccess = private)
        Capacity            % Maximum archive size
        M                   % Number of objectives
        Solutions = []      % Stored solutions
        Ages = []           % Age of each solution
        HypervolumeContributions = []  % HV contribution
        IdealPoint          % Ideal point estimation
        NadirPoint          % Nadir point estimation
        Size = 0            % Current archive size
    end
    
    methods
        %% Constructor
        function obj = SolutionArchive(capacity, M)
            obj.Capacity = capacity;
            obj.M = M;
            obj.IdealPoint = inf(1, M);
            obj.NadirPoint = -inf(1, M);
        end
        
        %% Add solutions to archive
        function add(obj, solutions)
            if isempty(solutions)
                return;
            end
            
            % Update ideal and nadir points
            objs = reshape([solutions.obj], obj.M, [])';
            obj.IdealPoint = min([obj.IdealPoint; objs], [], 1);
            obj.NadirPoint = max([obj.NadirPoint; objs], [], 1);
            
            % Combine with existing solutions
            if obj.Size > 0
                allSolutions = [obj.Solutions, solutions];
                allAges = [obj.Ages + 1, zeros(1, length(solutions))];
            else
                allSolutions = solutions;
                allAges = zeros(1, length(solutions));
            end
            
            % Non-dominated sorting
            allObjs = reshape([allSolutions.obj], obj.M, [])';
            [FrontNo, ~] = NDSort(allObjs, [], length(allSolutions));
            
            % Select non-dominated solutions (rank 1)
            ndMask = (FrontNo == 1);
            ndSolutions = allSolutions(ndMask);
            ndAges = allAges(ndMask);
            ndObjs = allObjs(ndMask, :);
            
            % If too many non-dominated solutions, select by hypervolume contribution
            if length(ndSolutions) > obj.Capacity
                % Calculate hypervolume contributions
                hvContrib = obj.CalculateHypervolumeContributions(ndObjs);
                
                % Combined score: hypervolume contribution / (age + epsilon)
                epsilon = 1e-6;
                scores = hvContrib ./ (ndAges + epsilon);
                
                % Select top solutions by score
                [~, idx] = sort(scores, 'descend');
                selectedIdx = idx(1:obj.Capacity);
                
                obj.Solutions = ndSolutions(selectedIdx);
                obj.Ages = ndAges(selectedIdx);
                obj.HypervolumeContributions = hvContrib(selectedIdx);
                obj.Size = obj.Capacity;
            else
                obj.Solutions = ndSolutions;
                obj.Ages = ndAges;
                if ~isempty(ndObjs)
                    obj.HypervolumeContributions = obj.CalculateHypervolumeContributions(ndObjs);
                end
                obj.Size = length(ndSolutions);
            end
        end
        
        %% Get training data with stratified sampling
        function data = getTrainingData(obj, n)
            if obj.Size == 0
                data = [];
                return;
            end
            
            n = min(n, obj.Size);
            
            % Stratified sampling based on hypervolume contribution
            probs = obj.HypervolumeContributions / sum(obj.HypervolumeContributions);
            
            % Ensure valid probabilities
            probs = max(probs, 1e-6);
            probs = probs / sum(probs);
            
            % Sample indices
            idx = randsample(obj.Size, n, true, probs);
            data = obj.Solutions(idx);
        end
        
        %% Generate reference points for conditioning
        function refPoints = getReferencePoints(obj, n)
            if obj.Size == 0 || isempty(obj.IdealPoint) || isempty(obj.NadirPoint)
                % Generate random reference points
                refPoints = rand(n, obj.M);
                return;
            end
            
            % Generate reference points on the simplex
            if obj.M == 2
                % 2D: evenly spaced points
                weights = linspace(0, 1, n)';
                refPoints = [weights, 1 - weights];
            elseif obj.M == 3
                % 3D: use simplex-lattice design
                p = ceil(sqrt(2 * n));
                refPoints = [];
                for i = 0:p
                    for j = 0:p-i
                        k = p - i - j;
                        refPoints = [refPoints; i/p, j/p, k/p];
                    end
                end
                % Select n points
                if size(refPoints, 1) > n
                    idx = randperm(size(refPoints, 1), n);
                    refPoints = refPoints(idx, :);
                end
            else
                % Higher dimensions: random weights normalized to sum to 1
                refPoints = rand(n, obj.M);
                refPoints = refPoints ./ sum(refPoints, 2);
            end
            
            % Scale to objective range
            range = obj.NadirPoint - obj.IdealPoint;
            refPoints = obj.IdealPoint + refPoints .* range;
        end
        
        %% Get ideal and nadir points
        function [ideal, nadir] = getBounds(obj)
            ideal = obj.IdealPoint;
            nadir = obj.NadirPoint;
        end
        
        %% Get all solutions
        function solutions = getAllSolutions(obj)
            solutions = obj.Solutions;
        end
    end
    
    methods (Access = private)
        %% Calculate hypervolume contributions
        function hvContrib = CalculateHypervolumeContributions(obj, objs)
            n = size(objs, 1);
            hvContrib = zeros(n, 1);
            
            if n == 0
                return;
            end
            
            % Use reference point slightly worse than nadir
            refPoint = obj.NadirPoint * 1.1;
            refPoint(refPoint < 0) = obj.NadirPoint(refPoint < 0) * 0.9;
            
            % Calculate total hypervolume
            totalHV = obj.ApproximateHypervolume(objs, refPoint);
            
            % Calculate contribution of each point
            for i = 1:n
                remaining = objs([1:i-1, i+1:end], :);
                hvWithout = obj.ApproximateHypervolume(remaining, refPoint);
                hvContrib(i) = max(0, totalHV - hvWithout);
            end
            
            % Normalize
            if sum(hvContrib) > 0
                hvContrib = hvContrib / sum(hvContrib);
            else
                hvContrib = ones(n, 1) / n;
            end
        end
        
        %% Approximate hypervolume using Monte Carlo
        function hv = ApproximateHypervolume(obj, objs, refPoint)
            if isempty(objs)
                hv = 0;
                return;
            end
            
            nSamples = 10000;
            
            % Sample points in the objective space
            samples = rand(nSamples, obj.M);
            for i = 1:obj.M
                samples(:, i) = obj.IdealPoint(i) + samples(:, i) * (refPoint(i) - obj.IdealPoint(i));
            end
            
            % Count dominated samples
            dominated = zeros(nSamples, 1);
            for i = 1:size(objs, 1)
                dominated = dominated | all(samples <= objs(i, :), 2);
            end
            
            % Calculate hypervolume
            volume = prod(refPoint - obj.IdealPoint);
            hv = sum(dominated) / nSamples * volume;
        end
    end
end
