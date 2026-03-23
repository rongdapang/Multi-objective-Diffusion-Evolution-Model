classdef DDD5_Static
    methods(Static)
        function Population = EnvironmentalSelection(Population, N)
            [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) + length(Last) > N
                CrowdDis = CrowdingDistance(Population(Last).objs, ones(1, length(Last)));
                [~, Rank] = sort(CrowdDis, 'descend');
                Next(Last(Rank(1:N-sum(Next)))) = true;
            else
                Next(Last) = true;
            end
            Population = Population(Next);
        end
    end
end