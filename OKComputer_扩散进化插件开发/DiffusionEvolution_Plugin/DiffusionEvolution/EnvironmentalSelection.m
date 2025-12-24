function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
%EnvironmentalSelection - Select individuals based on non-dominated sorting and crowding distance
%
%   [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
%   selects N individuals from the combined population using non-dominated
%   sorting and crowding distance as described in NSGA-II.
%
%   Input:
%       Population  - Combined parent and offspring population
%       N           - Population size for next generation
%
%   Output:
%       Population  - Selected population for next generation
%       FrontNo     - Front numbers of selected individuals
%       CrowdDis    - Crowding distances of selected individuals
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025. Part of PlatEMO framework.
%--------------------------------------------------------------------------

% Get objective values
Obj = [Population.objs];
M   = size(Obj,2);

% Non-dominated sorting
[FrontNo,MaxFrontNo] = NDSort(Obj,size(Obj,1));

% Calculate crowding distance
CrowdDis = CrowdingDistance(Obj,FrontNo);

% Environmental selection
if size(Population,1) <= N
    % If population size is less than or equal to N, return all individuals
    return;
else
    % Select individuals based on fronts and crowding distance
    Next = false(size(Population));
    FrontCount = 0;
    
    % Select complete fronts until we have enough individuals
    for i = 1:MaxFrontNo
        CurrentFront = find(FrontNo == i);
        if FrontCount + length(CurrentFront) <= N
            Next(CurrentFront) = true;
            FrontCount = FrontCount + length(CurrentFront);
        else
            % Sort the current front by crowding distance in descending order
            [~, SortOrder] = sort(-CrowdDis(CurrentFront));
            Remaining = N - FrontCount;
            Next(CurrentFront(SortOrder(1:Remaining))) = true;
            break;
        end
    end
    
    Population = Population(Next);
    FrontNo = FrontNo(Next);
    CrowdDis = CrowdDis(Next);
end

end

function [FrontNo,MaxFrontNo] = NDSort(Obj,PopNum)
%NDSort - Fast non-dominated sorting
%
%   [FrontNo,MaxFrontNo] = NDSort(Obj,PopNum)
%   performs fast non-dominated sorting on the objective values.
%
%   Input:
%       Obj     - Objective values matrix (PopNum x M)
%       PopNum  - Population size
%
%   Output:
%       FrontNo - Front numbers for each individual
%       MaxFrontNo - Maximum front number

% Initialize
FrontNo = zeros(1,PopNum);
MaxFrontNo = 0;

% Dominated counts and dominating sets
DominatedCount = zeros(1,PopNum);
DominateSet = cell(PopNum,1);

% First front
CurrentFront = false(1,PopNum);

% Compare each individual with all others
for i = 1:PopNum
    for j = i+1:PopNum
        % Check if i dominates j
        if all(Obj(i,:) <= Obj(j,:)) && any(Obj(i,:) < Obj(j,:))
            DominateSet{i}{end+1} = j;
            DominatedCount(j) = DominatedCount(j) + 1;
        % Check if j dominates i
        elseif all(Obj(j,:) <= Obj(i,:)) && any(Obj(j,:) < Obj(i,:))
            DominateSet{j}{end+1} = i;
            DominatedCount(i) = DominatedCount(i) + 1;
        end
    end
    % If not dominated by anyone, it's in the first front
    if DominatedCount(i) == 0
        CurrentFront(i) = true;
        FrontNo(i) = 1;
    end
end

MaxFrontNo = 1;

% Find subsequent fronts
while any(CurrentFront)
    NextFront = false(1,PopNum);
    for i = find(CurrentFront)'
        if ~isempty(DominateSet{i})
            for j = DominateSet{i}
                DominatedCount(j) = DominatedCount(j) - 1;
                if DominatedCount(j) == 0
                    NextFront(j) = true;
                    FrontNo(j) = MaxFrontNo + 1;
                end
            end
        end
    end
    CurrentFront = NextFront;
    MaxFrontNo = MaxFrontNo + 1;
end

end

function CrowdDis = CrowdingDistance(Obj,FrontNo)
%CrowdingDistance - Calculate crowding distance
%
%   CrowdDis = CrowdingDistance(Obj,FrontNo)
%   calculates the crowding distance for each individual.
%
%   Input:
%       Obj     - Objective values matrix
%       FrontNo - Front numbers
%
%   Output:
%       CrowdDis - Crowding distances

PopNum = size(Obj,1);
M = size(Obj,2);
CrowdDis = zeros(1,PopNum);

% Calculate crowding distance for each front
MaxFrontNo = max(FrontNo);
for Front = 1:MaxFrontNo
    FrontIndex = find(FrontNo == Front);
    FrontSize = length(FrontIndex);
    
    if FrontSize <= 2
        CrowdDis(FrontIndex) = inf;
    else
        % Calculate distance for each objective
        for m = 1:M
            % Sort by objective m
            [SortedObj, SortOrder] = sort(Obj(FrontIndex,m));
            
            % Boundary individuals have infinite distance
            CrowdDis(FrontIndex(SortOrder(1))) = inf;
            CrowdDis(FrontIndex(SortOrder(end))) = inf;
            
            % Calculate crowding distance for interior individuals
            if FrontSize > 2
                for i = 2:FrontSize-1
                    CrowdDis(FrontIndex(SortOrder(i))) = CrowdDis(FrontIndex(SortOrder(i))) + ...
                        (SortedObj(i+1) - SortedObj(i-1)) / (max(SortedObj) - min(SortedObj) + eps);
                end
            end
        end
    end
end

end