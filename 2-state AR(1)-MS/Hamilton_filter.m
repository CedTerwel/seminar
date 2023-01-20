function [ filteredxi , predictedxi ] = Hamilton_filter(p11,p22,mu,sigma,y)

% Extract length of data
T = size(y,2);

% Build transition matrix from p11 and p22
P   = [ p11 , 1-p22 ; 1-p11 , p22];

% Run the Hamilton filter
for t = 1:T
    
    % PREDICTION STEP
    if t==1
        % Initialise predictedxi(:,1)
        %predictedxi(:,t) = P*[ 1 ; 0 ]; % start in state 1
        %predictedxi(:,t) = P*[ 0 ; 1 ]; % start in state 2
        predictedxi(:,t) = [ (1-p22) / (2-p11-p22) ; (1-p11)/(2-p11-p22) ]; % unconditional probability
    else
        predictedxi(:,t)= P * filteredxi(:,t-1);
    end
    
    % UPDATE STEP
    likelihood(:,t)   = [ normpdf(y(1,t),mu(1),sigma(1)) ; normpdf(y(1,t),mu(2),sigma(2)) ];
    filteredxi(:,t)   = predictedxi(:,t) .* likelihood(:,t) / ([1,1]*(predictedxi(:,t).*likelihood(:,t)) );
   
end

end % Close the function


