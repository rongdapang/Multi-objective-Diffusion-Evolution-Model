function Offspring = OperatorGA(Problem,Parent)
%OperatorGA - Apply genetic operators (crossover and mutation)
%
%   Offspring = OperatorGA(Problem,Parent)
%   applies simulated binary crossover (SBX) and polynomial mutation
%   to generate offspring.
%
%   Input:
%       Problem - Problem instance
%       Parent  - Parent population
%
%   Output:
%       Offspring - Generated offspring
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025. Part of PlatEMO framework.
%--------------------------------------------------------------------------

% Parameters
Pc = 1;  % Crossover probability
Pm = 1/Problem.D;  % Mutation probability
EtaC = 20;  % Crossover distribution index
EtaM = 20;  % Mutation distribution index

% Initialize offspring
Offspring = repmat(struct('decision',[],'objs',[],'cons',[]),size(Parent,1),1);

% Apply crossover and mutation
for i = 1:2:length(Parent)
    % Random pairing
    if i+1 <= length(Parent)
        Parent1 = Parent(i);
        Parent2 = Parent(i+1);
        
        % Crossover
        if rand() < Pc
            [Off1,Off2] = SBX(Parent1.decision,Parent2.decision,Problem.lower,Problem.upper,EtaC);
        else
            Off1 = Parent1.decision;
            Off2 = Parent2.decision;
        end
        
        % Mutation
        Off1 = PolynomialMutation(Off1,Problem.lower,Problem.upper,Pm,EtaM);
        Off2 = PolynomialMutation(Off2,Problem.lower,Problem.upper,Pm,EtaM);
        
        % Store offspring
        Offspring(i).decision = Off1;
        if i+1 <= length(Offspring)
            Offspring(i+1).decision = Off2;
        end
    else
        % Last individual if odd number
        Offspring(i).decision = Parent(i).decision;
    end
end

% Evaluate offspring
Offspring = Problem.Evaluation(Offspring);

end

function [Off1,Off2] = SBX(Parent1,Parent2,Lower,Upper,EtaC)
%SBX - Simulated binary crossover
%
%   [Off1,Off2] = SBX(Parent1,Parent2,Lower,Upper,EtaC)
%   performs simulated binary crossover.

D = length(Parent1);
Off1 = Parent1;
Off2 = Parent2;

for i = 1:D
    if rand() <= 0.5
        if abs(Parent1(i) - Parent2(i)) > eps
            % Calculate beta
            if Parent1(i) < Parent2(i)
                y1 = Parent1(i);
                y2 = Parent2(i);
            else
                y1 = Parent2(i);
                y2 = Parent1(i);
            end
            
            yl = Lower(i);
            yu = Upper(i);
            
            beta = 1 + (2 * (y1 - yl) / (y2 - y1));
            alpha = 2 - beta^(-(EtaC + 1));
            
            if alpha < 0
                betaq = (2 * rand())^(1 / (EtaC + 1));
            else
                if rand() <= (1 / alpha)
                    betaq = (2 * rand())^(1 / (EtaC + 1));
                else
                    betaq = (1 / (2 - 2 * rand()))^(1 / (EtaC + 1));
                end
            end
            
            % Generate offspring
            Off1(i) = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
            Off2(i) = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
            
            % Repair bounds
            Off1(i) = max(Off1(i), Lower(i));
            Off1(i) = min(Off1(i), Upper(i));
            Off2(i) = max(Off2(i), Lower(i));
            Off2(i) = min(Off2(i), Upper(i));
        end
    end
end

end

function Offspring = PolynomialMutation(Parent,Lower,Upper,Pm,EtaM)
%PolynomialMutation - Polynomial mutation operator
%
%   Offspring = PolynomialMutation(Parent,Lower,Upper,Pm,EtaM)
%   applies polynomial mutation to the parent.

D = length(Parent);
Offspring = Parent;

for i = 1:D
    if rand() < Pm
        y = Parent(i);
        yl = Lower(i);
        yu = Upper(i);
        
        delta1 = (y - yl) / (yu - yl);
        delta2 = (yu - y) / (yu - yl);
        
        mut_pow = 1 / (EtaM + 1);
        
        if rand() <= 0.5
            xy = 1 - delta1;
            if xy < 0
                val = 0;
            else
                val = 2 * rand() + (1 - 2 * rand()) * (1 - xy)^(EtaM + 1);
                if val < 0
                    deltaq = val^mut_pow - 1;
                else
                    deltaq = 1 - val^mut_pow;
                end
            end
        else
            xy = 1 - delta2;
            if xy < 0
                val = 0;
            else
                val = 2 * (1 - rand()) + 2 * (rand() - 0.5) * (1 - xy)^(EtaM + 1);
                if val < 0
                    deltaq = val^mut_pow - 1;
                else
                    deltaq = 1 - val^mut_pow;
                end
            end
        end
        
        y = y + deltaq * (yu - yl);
        y = max(y, yl);
        y = min(y, yu);
        Offspring(i) = y;
    end
end

end