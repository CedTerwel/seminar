clear all

format bank

% load data.mat
% data = table2array(data);

% noEMiterations = 1000; % how many iterations of the EM algorithm
% printEMupdates = 0; % toggle EM updates on (1) or off (0 - or any other value, really)
% 
% y = data(:,3)';
% T = size(y,2);

load ccreturns_clean.mat
%data = table2array(data);

noEMiterations = 1000; % how many iterations of the EM algorithm
printEMupdates = 0; % toggle EM updates on (1) or off (0 - or any other value, really)

%y = data(:,3)';
y = ccreturns_clean(2:end)';
T = size(y,2);


%% === Maximum Likelihood ===

% Starting values based roughly on the data
% starting values = [p_{11},p_{22},[mu_1;mu_2],[sigma_1;sigma_2] ]
%startingvalues = [0.8;0.8;[mean(y);mean(y)];[1/2*std(y);2*std(y)]];

% starting values = [p_{11},p_{21},p_{31},p_{22},p_{33},p_{14},p_{24},p_{34},
%                    [mu_1;mu_2;mu_3;mu_4],
%                    [sigma_1;sigma_2];sigma_3;sigma_4]];
startingvalues = [0.25;0.25;0.25;0.5;0.5;0.25;0.25;0.25;
                  [mean(y);mean(y);mean(y);mean(y)];
                  [2*std(y);std(y);1/2*std(y);1/4*std(y)]];


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

% lb = [0;0;-Inf;-Inf;0;0]; % Lower bound for the parameter vector [p_{11},p_{22},mu_1,mu_2,sigma_1,sigma_2]
% ub = [1;1;Inf;Inf;Inf;Inf]; % Upper bound
lb = [0;0;0;0;0;0;0;0;-Inf; 0  ;-Inf;  0 ; 0 ; 0 ; 0 ; 0 ]; % Lower bound for the parameter vector [p_{11},p_{22},mu_1,mu_2,sigma_1,sigma_2]
ub = [1;1;1;1;1;1;1;1; 0  ; Inf; 0  ; Inf;Inf;Inf;Inf;Inf]; % Upper bound

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
% p11,p21, p31, p22, p33, p14, p24, p34,
p11   = parameters_ML1(1,1);
p21   = parameters_ML1(2,1);
p31   = parameters_ML1(3,1);
p22   = parameters_ML1(4,1);
p33   = parameters_ML1(5,1);
p14   = parameters_ML1(6,1);
p24   = parameters_ML1(7,1);
p34   = parameters_ML1(8,1);
mu    = parameters_ML1(9:12,1);
sigma = parameters_ML1(13:16,1);
[ filteredxi , predictedxi ] = Hamilton_filter(p11,p21,p31,p22,p33,p14,p24,p34,mu,sigma,y);
[ smoothedxi ,xi0_out , Pstar ] = Hamilton_smoother(p11,p21,p31,p22,p33,p14,p24,p34,mu,sigma,y);
mu;

% figure
% plot(y,'k','Linewidth',0.3)
% hold on
% plot(predictedxi(2,:),'c','Linewidth',2)
% hold on
% plot(filteredxi(2,:),'g','Linewidth',2)
% hold on
% plot(smoothedxi(2,:),'b:','Linewidth',2)
% hold on
% axis([0 inf -8 8])
% %axis([0 inf -1 1])
% hold off
% set(gca,'FontSize',14)
% set(gca,'FontName','Times New Roman')
% legend('Data','Predicted state','Filtered state','Smoothed state')
% legend('Data','Smoothed state')

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