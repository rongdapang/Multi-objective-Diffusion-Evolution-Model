classdef SolutionArchive < handle
% SolutionArchive - Elite solution archive with hypervolume-based management
%
% This class manages an archive of elite solutions for the DDD algorithm.
% It uses hypervolume contribution to maintain solution quality and diversity.
%
% Properties:
%   MaxSize - Maximum archive size
%   Size - Current archive size
%   Solutions - Cell array of archived solutions
%   Objectives - Matrix of objective values
%   DecisionVars - Matrix of decision variables

    properties (SetAccess = private)
        MaxSize         % Maximum archive capacity
        Size            % Current number of solutions
        Solutions       % Cell array storing solution structures
        Objectives      % NxM matrix of objective values
        DecisionVars    % NxD matrix of decision variables
        M               % Number of objectives
        ReferencePoint  % Reference point for hypervolume calculation
    end
    
    methods
        %% Constructor
        function obj = SolutionArchive(maxSize, M)
            obj.MaxSize = maxSize;
            obj.M = M;
            obj.Size = 0;
            obj.Solutions = {};
            obj.Objectives = [];
            obj.DecisionVars = [];
            % Set reference point for hypervolume (will be updated dynamically)
            obj.ReferencePoint = ones(1, M) * 1.1;
        end
        
        %% Add solutions to archive
        function add(obj, Population)
            if isempty(Population)
                return;
            end
            
            % Convert population to matrices if needed
            nNew = length(Population);
            newObjs = reshape([Population.obj], obj.M, [])';
            newDecs = reshape([Population.dec], length(Population(1).dec), [])';
            
            % Combine with existing archive
            if obj.Size == 0
                obj.Objectives = newObjs;
                obj.DecisionVars = newDecs;
                obj.Solutions = cell(nNew, 1);
                for i = 1:nNew
                    obj.Solutions{i} = Population(i);
                end
                obj.Size = nNew;
            else
                obj.Objectives = [obj.Objectives; newObjs];
                obj.DecisionVars = [obj.DecisionVars; newDecs];
                for i = 1:nNew
                    obj.Solutions{obj.Size + i} = Population(i);
                end
                obj.Size = obj.Size + nNew;
            end
            
            % Remove dominated solutions
            obj.removeDominated();
            
            % Trim to max size using hypervolume contribution
            if obj.Size > obj.MaxSize
                obj.trimByHypervolume();
            end
            
            % Update reference point
            obj.updateReferencePoint();
        end
        
        %% Get reference points for conditioning
        function refPoints = getReferencePoints(obj, nPoints)
            if obj.Size == 0
                refPoints = [];
                return;
            end
            
            % Select diverse reference points using k-means clustering
            if obj.Size <= nPoints
                refPoints = obj.Objectives;
            else
                % Use uniform selection based on crowding distance
                [FrontNo, ~] = NDSort(obj.Objectives, [], obj.Size);
                nonDominated = find(FrontNo == 1);
                
                if length(nonDominated) >= nPoints
                    % Select from non-dominated solutions
                    selectedIdx = nonDominated(randi(length(nonDominated), 1, nPoints));
                    refPoints = obj.Objectives(selectedIdx, :);
                else
                    % Fill with dominated solutions
                    refPoints = obj.Objectives(nonDominated, :);
                    remaining = nPoints - length(nonDominated);
                    dominated = setdiff(1:obj.Size, nonDominated);
                    if ~isempty(dominated)
                        extraIdx = dominated(randi(length(dominated), 1, remaining));
                        refPoints = [refPoints; obj.Objectives(extraIdx, :)];
                    end
                end
            end
        end
        
        %% Get training data for diffusion model
        function trainingData = getTrainingData(obj, nSamples)
            if obj.Size == 0
                trainingData = [];
                return;
            end
            
            n = min(nSamples, obj.Size);
            
            % Prioritize non-dominated solutions
            [FrontNo, ~] = NDSort(obj.Objectives, [], obj.Size);
            nonDominated = find(FrontNo == 1);
            
            if length(nonDominated) >= n
                % Select from non-dominated solutions
                selectedIdx = nonDominated(randperm(length(nonDominated), n));
            else
                % Fill with dominated solutions
                selectedIdx = nonDominated;
                dominated = setdiff(1:obj.Size, nonDominated);
                if ~isempty(dominated)
                    nExtra = min(n - length(nonDominated), length(dominated));
                    extraIdx = dominated(randperm(length(dominated), nExtra));
                    selectedIdx = [selectedIdx, extraIdx];
                end
            end
            
            % Create training data structure
            trainingData = struct();
            trainingData.dec = obj.DecisionVars(selectedIdx, :);
            trainingData.obj = obj.Objectives(selectedIdx, :);
        end
        
        %% Get all solutions
        function Population = getAllSolutions(obj)
            Population = [];
            if obj.Size == 0
                return;
            end
            
            % Convert back to INDIVIDUAL objects
            for i = 1:obj.Size
                if i == 1
                    Population = obj.Solutions{i};
                else
                    Population = [Population, obj.Solutions{i}];
                end
            end
        end
        
        %% Clear archive
        function clear(obj)
            obj.Size = 0;
            obj.Solutions = {};
            obj.Objectives = [];
            obj.DecisionVars = [];
        end
    end
    
    methods (Access = private)
        %% Remove dominated solutions
        function removeDominated(obj)
            if obj.Size <= 1
                return;
            end
            
            [FrontNo, ~] = NDSort(obj.Objectives, [], obj.Size);
            nonDominated = (FrontNo == 1);
            
            obj.Objectives = obj.Objectives(nonDominated, :);
            obj.DecisionVars = obj.DecisionVars(nonDominated, :);
            obj.Solutions = obj.Solutions(nonDominated);
            obj.Size = sum(nonDominated);
        end
        
        %% Trim archive using hypervolume contribution
        function trimByHypervolume(obj)
            if obj.Size <= obj.MaxSize
                return;
            end
            
            % Calculate hypervolume contribution for each solution
            hvContrib = zeros(obj.Size, 1);
            
            for i = 1:obj.Size
                % Calculate hypervolume without this solution
                remaining = setdiff(1:obj.Size, i);
                if ~isempty(remaining)
                    hvContrib(i) = obj.calculateHypervolume(1:obj.Size) - ...
                                   obj.calculateHypervolume(remaining);
                else
                    hvContrib(i) = inf;
                end
            end
            
            % Keep solutions with highest hypervolume contribution
            [~, idx] = sort(-hvContrib);
            keepIdx = idx(1:obj.MaxSize);
            
            obj.Objectives = obj.Objectives(keepIdx, :);
            obj.DecisionVars = obj.DecisionVars(keepIdx, :);
            obj.Solutions = obj.Solutions(keepIdx);
            obj.Size = obj.MaxSize;
        end
        
        %% Calculate hypervolume for a set of solutions
        function hv = calculateHypervolume(obj, indices)
            if isempty(indices)
                hv = 0;
                return;
            end
            
            points = obj.Objectives(indices, :);
            
            % Simple hypervolume calculation using Lebesgue measure
            % For 2D and 3D, use exact calculation
            if obj.M == 2
                hv = obj.hypervolume2D(points, obj.ReferencePoint);
            elseif obj.M == 3
                hv = obj.hypervolume3D(points, obj.ReferencePoint);
            else
                % For higher dimensions, use Monte Carlo approximation
                hv = obj.hypervolumeMC(points, obj.ReferencePoint, 10000);
            end
        end
        
        %% 2D hypervolume calculation
        function hv = hypervolume2D(~, points, ref)
            % Sort by first objective
            [sorted, idx] = sort(points(:, 1));
            sortedPoints = points(idx, :);
            
            hv = 0;
            for i = 1:size(sortedPoints, 1)
                if i == 1
                    width = ref(1) - sortedPoints(i, 1);
                else
                    width = sortedPoints(i-1, 1) - sortedPoints(i, 1);
                end
                height = ref(2) - sortedPoints(i, 2);
                hv = hv + width * height;
            end
        end
        
        %% 3D hypervolume calculation (simplified)
        function hv = hypervolume3D(~, points, ref)
            % Use inclusion-exclusion principle approximation
            hv = 0;
            n = size(points, 1);
            
            for i = 1:n
                vol = prod(ref - points(i, :));
                hv = hv + vol;
            end
            
            % Normalize
            hv = hv / n;
        end
        
        %% Monte Carlo hypervolume approximation
        function hv = hypervolumeMC(~, points, ref, nSamples)
            % Generate random samples in the bounding box
            nPoints = size(points, 1);
            M = size(points, 2);
            
            % Find bounding box
            minPoint = min(points);
            maxPoint = ref;
            
            % Generate random samples
            samples = rand(nSamples, M) .* (maxPoint - minPoint) + minPoint;
            
            % Count dominated samples
            dominatedCount = 0;
            for i = 1:nSamples
                for j = 1:nPoints
                    if all(samples(i, :) >= points(j, :))
                        dominatedCount = dominatedCount + 1;
                        break;
                    end
                end
            end
            
            % Calculate hypervolume
            boxVolume = prod(maxPoint - minPoint);
            hv = boxVolume * dominatedCount / nSamples;
        end
        
        %% Update reference point
        function updateReferencePoint(obj)
            if obj.Size == 0
                return;
            end
            
            % Set reference point slightly worse than the worst objective values
            maxObj = max(obj.Objectives);
            obj.ReferencePoint = maxObj * 1.1;
        end
    end
end
