function [ filteredxi , predictedxi ] = Hamilton_filter(p11,p21, p31, p22, p33, p14, p24, p34, mu,sigma,y)

% Extract length of data
T = size(y,2);

% Build transition matrix from p11 and p22
%P   = [ p11, 1-p22 ; 1-p11 , p22];
P   = [ p11                 , 1 - p22 , 0       , p14;
        p21                 , p22     , 0       , p24;
        p31                 , 0       , p33     , p34;
        1 - p11 - p21 - p31 , 0       , 1 - p33 , 1 - p14 - p24 - p34];
P = P';

% Run the Hamilton filter
for t = 1:T
    
    % PREDICTION STEP
    if t==1
        % Initialise predictedxi(:,1)
        %predictedxi(:,t) = P*[ 1 ; 0 ]; % start in state 1
        %predictedxi(:,t) = P*[ 0 ; 1 ]; % start in state 2
        %predictedxi(:,t) = [ (1-p22) / (2-p11-p22) ; (1-p11)/(2-p11-p22) ]; % unconditional probability
        % Determine eigenvalues with det(P-\lambda * I) = 0, than use
        % \lambda in solving (P - \lambda*I)v = 0, where v is the
        % eigenvector
        predictedxi(:,t) = [ 0.25 ; 0.25 ; 0.25 ; 0.25 ];
        
    else
        predictedxi(:,t)= P * filteredxi(:,t-1);
    end
    
    % UPDATE STEP
    likelihood(:,t)   = [ normpdf(y(1,t),mu(1),sigma(1)) ; 
                          normpdf(y(1,t),mu(2),sigma(2)) ;
                          normpdf(y(1,t),mu(3),sigma(3)) ; 
                          normpdf(y(1,t),mu(4),sigma(4)) ];
    filteredxi(:,t)   = predictedxi(:,t) .* likelihood(:,t) / ([1,1,1,1]*(predictedxi(:,t).*likelihood(:,t)) );
   
end

end % Close the function


