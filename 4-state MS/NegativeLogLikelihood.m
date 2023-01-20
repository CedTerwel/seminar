function [output]=NegativeLogLikelihood(parameter_vector,y)

% Extract the stuff we need from the input arguments
p11   = parameter_vector(1,1);
p21   = parameter_vector(2,1);
p31   = parameter_vector(3,1);
p22   = parameter_vector(4,1);
p33   = parameter_vector(5,1);
p14   = parameter_vector(6,1);
p24   = parameter_vector(7,1);
p34   = parameter_vector(8,1);
mu    = [ parameter_vector(9,1) ; parameter_vector(10,1) ; parameter_vector(11,1) ; parameter_vector(12,1) ];
sigma = [ parameter_vector(13,1) ; parameter_vector(14,1); parameter_vector(15,1) ; parameter_vector(16,1) ];

% Run the Hamilton filter
[~,predictedxi] = Hamilton_filter(p11,p21, p31, p22, p33, p14, p24, p34, mu,sigma,y);


% Collect a row vector of log likelihood per observation
LL =  log( predictedxi(1,:) .* normpdf( y , mu(1) , sigma(1) ) + ... 
           predictedxi(2,:) .* normpdf( y , mu(2) , sigma(2) ) + ...
           predictedxi(3,:) .* normpdf( y , mu(3) , sigma(3) ) + ...
           predictedxi(4,:) .* normpdf( y , mu(4) , sigma(4) ));

% Sum over all observations and put a minus sign in front
output = - sum( LL(1:end) );               

end

