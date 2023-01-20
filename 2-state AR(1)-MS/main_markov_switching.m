clear all

format bank

load ccreturns_clean.mat
%data = table2array(data);

noEMiterations = 1000; % how many iterations of the EM algorithm
printEMupdates = 0; % toggle EM updates on (1) or off (0 - or any other value, really)

y = ccreturns_clean(1:end)';
T = size(y,2);


%% === Maximum Likelihood ===

% Starting values based roughly on the data
% starting values = [p_{11},p_{22},[mu_1;mu_2],[beta_1;beta_2],[sigma_1;sigma_2] ]
startingvalues = [0.8;0.8;[mean(y);mean(y)];[0.9; 0.9];[1/2*std(y);2*std(y)]];

% It is useful to take the starting points *not* symmetric in sigma_1 and
% sigma_2. Also, it may help to initialise the p_{11} and p_{22} as relatively persistent

clearvars options
options  =  optimset('fminunc'); % This sets the options at their standard values
options  =  optimset(options, 'Display','off');
options  =  optimset(options , 'MaxFunEvals' , 10^6) ; % extra iterations
options  =  optimset(options , 'TolFun'      , 1e-6); % extra precision
options  =  optimset(options , 'TolX'        , 1e-6); % extra precision

% Run the ML optimisation and store the result:
[parameters_ML1,LogL_ML1] = fminunc('NegativeLogLikelihood',startingvalues,options,y);

%% Optimise using fmincon ===

clearvars options
options  =  optimset('fmincon'); % This sets the options at their standard values
options  =  optimset(options, 'Display','off');
options  =  optimset(options , 'MaxFunEvals' , 10^6) ; % extra iterations
options  =  optimset(options , 'TolFun'      , 1e-6); % extra precision
options  =  optimset(options , 'TolX'        , 1e-6); % extra precision

lb = [0;0;-100;-100;-100;-100;0;0]; % Lower bound for the parameter vector [p_{11},p_{22},mu_1,mu_2,beta_1,beta_2sigma_1,sigma_2]
ub = [1;1;100;100;100;100;100;100]; % Upper bound

[parameters_ML2,LogL_ML2] = fmincon('NegativeLogLikelihood', startingvalues,[],[],[],[],lb,ub,[],options,y);

%% Calculate standard errors and display results both optimisers

% Standard errors are calculated by calculating the Hessian of the
% LogLikelihood at the ML parameters. Then invert, take diagional, square
% root.

%format short
ML_se = sqrt( abs( diag ( inv( fdhess6('NegativeLogLikelihood',parameters_ML1,y) ))));

% display true parameters, estimated, and standard error
format short
disp("Parameters ML1, ML2, and Std dev")
disp([parameters_ML1,parameters_ML2,ML_se])

NegativeLogLikelihood(parameters_ML1,y)

%% Plot predicted, filtered and smoothed states
p11   = parameters_ML1(1,1);
p22   = parameters_ML1(2,1);
mu    = parameters_ML1(3:4,1);
beta  = parameters_ML1(5:6,1);
sigma = parameters_ML1(7:8,1);
[ filteredxi , predictedxi ] = Hamilton_filter(p11,p22,mu,beta,sigma,y);
[ smoothedxi ,xi0_out , Pstar ] = Hamilton_smoother(p11,p22,mu,beta,sigma,y);

figure
plot(y,'k','Linewidth',0.3)
hold on
% plot(predictedxi(2,:),'c','Linewidth',2)
% hold on
% plot(filteredxi(2,:),'g','Linewidth',2)
% hold on
plot(smoothedxi(2,:),'b:','Linewidth',2)
hold on
axis([0 inf -8 8])
%axis([0 inf -1 1])
hold off
set(gca,'FontSize',14)
set(gca,'FontName','Times New Roman')
%legend('Data','Predicted state','Filtered state','Smoothed state')
legend('Data','Smoothed state')

%% Make 1- and 2-step ahead forecasts
% Forecast xi
P   = [ p11 , 1-p22 ; 1-p11 , p22];
fcast1xi = P*smoothedxi(:,end);
fcast2xi = (P^2)*smoothedxi(:,end);

% Forecast y and calculate forecast errors
fcastY = [fcast1xi'*mu, fcast2xi'*mu];
fcastError = [y190-fcastY(1,1), y191-fcastY(1,2)];

% %% === EM Algorithm ===
% 
% % Initialize parameter values for first EM run
% p11   = 0.8;
% p22   = 0.8;
% mu1   = mean(y);
% mu2   = mean(y);
% sigma1= 0.5*std(y);
% sigma2= 1.5*std(y);
% xi0_in = [0.5;0.5];
% 
% %% Iterations
% iteration=0; % counter for iterations
% 
% format long
% 
% parameters_EM = nan(8,noEMiterations);
% LogL_EM = nan(1,noEMiterations);
% 
% for iter = 1:noEMiterations
%   % Run EM step
%   [p11,p22,mu1,mu2,sigma1,sigma2,xi0_out] = EM_step(p11,p22,mu1,mu2,sigma1,sigma2,xi0_in,y);
%   
%   % Set in=out to make sure we update:
%   xi0_in = xi0_out;
%   
%   % Save the parameters
%   parameters_EM(:,iter) = [p11;p22;mu1;mu2;sigma1;sigma2;xi0_out];
%   
%   % print the latest parameters
%   if printEMupdates == 1
%     disp(parameters_EM(:,iter))
%   end
%   
%   % Calculate the log-likelihood at these parameters
%   LogL_EM(iter) = - NegativeLogLikelihood2(parameters_EM(:,iter),y);
%   
%  end
% 
% %% Ideally, both the likelihood and the parameters should converge pretty quickly
% figure
% plot(LogL_EM,'LineWidth',2);
% 
% figure
% plot(parameters_EM(1:6,:)','LineWidth',2);
% 
% 
% %% Compare the obtained parameters by ML and EM
% format short
% long_term_average =  [(1-p22) / (2-p11-p22);(1-p11)/(2-p11-p22)];
% 
% disp("Parameter and long term average for ML1, ML2 and EM")
% disp([[parameters_ML1;long_term_average],[parameters_ML2;long_term_average],parameters_EM(:,end)])
% 
% %% Compare the obtained log likelihoods
% format bank
% disp("Log likelihoods: ML1, ML2, EM")
% disp([NegativeLogLikelihood(parameters_ML1,y),NegativeLogLikelihood(parameters_ML2,y),NegativeLogLikelihood2(parameters_EM(:,end),y)])