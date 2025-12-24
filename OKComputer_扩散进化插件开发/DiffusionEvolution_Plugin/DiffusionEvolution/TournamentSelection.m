function MatingPool = TournamentSelection(k,N,FrontNo,CrowdDis)
%TournamentSelection - Select individuals using tournament selection
%
%   MatingPool = TournamentSelection(k,N,FrontNo,CrowdDis)
%   performs tournament selection to select mating pool.
%
%   Input:
%       k        - Tournament size
%       N        - Number of selections
%       FrontNo  - Front numbers for each individual
%       CrowdDis - Crowding distances (negative for minimization)
%
%   Output:
%       MatingPool - Selected indices
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025. Part of PlatEMO framework.
%--------------------------------------------------------------------------

PopNum = length(FrontNo);
MatingPool = zeros(1,N);

for i = 1:N
    % Randomly select k individuals for tournament
    Candidate = randi(PopNum,k,1);
    
    % Find the best individual in the tournament
    [Best,~] = min(FrontNo(Candidate));
    BestCandidate = Candidate(FrontNo(Candidate) == Best);
    
    % If multiple individuals are in the same front, use crowding distance
    if length(BestCandidate) > 1
        [~,BestIdx] = max(CrowdDis(BestCandidate));
        BestCandidate = BestCandidate(BestIdx);
    end
    
    MatingPool(i) = BestCandidate(1);
end

end