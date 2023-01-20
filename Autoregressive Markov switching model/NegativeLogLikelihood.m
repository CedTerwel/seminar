function [output]=NegativeLogLikelihood(parameter_vector,y)

% Extract the stuff we need from the input arguments
p11   = parameter_vector(1,1);
p22   = parameter_vector(2,1);
mu    = [ parameter_vector(3,1) ; parameter_vector(4,1) ];
beta  = [ parameter_vector(5,1) ; parameter_vector(6,1) ];
sigma = [ parameter_vector(7,1) ; parameter_vector(8,1) ];

% Run the Hamilton filter
[~,predictedxi] = Hamilton_filter(p11,p22,mu,beta,sigma,y);

% Collect a row vector of log likelihood per observation
for t = 2:length(y)
    LL(t) =  log( predictedxi(1,t) * normpdf( y(1,t), mu(1)+beta(1)*y(1,t-1), sigma(1)) + predictedxi(2,t) * normpdf( y(1,t), mu(2)+beta(2)*y(1,t-1), sigma(2) ) );
end

% Sum over all observations and put a minus sign in front
output = - sum( LL(1:end) );               

end

