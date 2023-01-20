function [ smoothedxi ,xi0_out , Pstar ] = Hamilton_smoother(p11,p22,mu,sigma,y)
  
  % Extract length of data
  T = size(y,2);
  
  % Build transition matrix from p11 and p22
  P   = [ p11 , 1-p22 ; 1-p11 , p22];
  
  % Run the Hamilton filter
  [ filteredxi , predictedxi ] = Hamilton_filter(p11,p22,mu,sigma,y);
  
  % Initialise the smoother at time T
  smoothedxi = nan(size(filteredxi,1),T);
  smoothedxi(:,T) = filteredxi(:,T);
  
  % Run the `Kim smoother' backwards
  for t = T-1:-1:1 %i = 1:(T-1)
    %t = T - i;
    smoothedxi(:,t) = filteredxi(:,t) .* ( P' * ( smoothedxi(:,t+1) ./ predictedxi(:,t+1) ) );
  end
  % Notice that in Matlab, .* is elementwise multiplication and ./ is
  % elementwise division. Normal matrix multiplication is given by *
  
  % This is what we though originally:
  xi0_in = P*[ 1 ; 0 ];
  %xi0_in = P*[ 0 ; 1 ];
  %xi0_in = [ (1-p22) / (2-p11-p22) ; (1-p11)/(2-p11-p22) ];
  
  % But now we can get a smoothed estimate of xi0
  xi0_out = xi0_in .* ( P' * ( smoothedxi(:,1) ./ predictedxi(:,1) ) );
  
  %Get the cross terms
  Pstar = nan(2,2,T);
  Pstar(:,:,1) = P.* (smoothedxi(:,1) * xi0_in') ./ ( predictedxi(:,1) * [1,1]);
  for t = 2:T
    Pstar(:,:,t) = P.* (smoothedxi(:,t) * filteredxi(:,t-1)') ./ ( predictedxi(:,t) * [1,1]) ;
  end
  
end

