% Run file for an example linear regression model
% programmer: me =]
%
% data: data1.csv is a 20x3 matrix containing randomly generated numbers
%		make sure to save your csv file format as specialty .csv, the common UTF-8 will not work
%
% goal: given X matrix of m by n, m examples and n features, and a vector y with m results,
%       use linear regression to find theta that best fit the given data.

% setup and house keeping
clear; close all; clc

% load data
data = load('data1.csv');
X = data(:, 1:2);              	% columns 1 and 2 are input parameters
y = data(:, 3);                	% column 3 is our label
m = length(y);                 	% get number of rows

% explore data
plot(X(:,1), y, 'bx', 'MarkerSize', 8);
print -djpg 'fig1_X_col1.jpg';                             % save figure 1
% if you are receiving fig2dev warning on the print function, you will need to install the package
% or turn the warning off, the figure will still save to your directory

% normalize X which will help converges quickly
X_norm = zeros(size(X));        % initialize X_norm
mu = zeros(1, size(X,2));       % initialize mu for mean per column
sigma = zeros(1, size(X,2));    % initialize sigma for std per column

for c=1:size(X,2),
	cmean = mean(X(:,c));       % get column mean
	cstd = std(X(:,c));         % get column std
	for r=1:size(X,1),
	    X_norm(r,c) = (X(r,c) - cmean) / cstd;             % store normalized values
	end;
	mu(1,c) = cmean;            % store column mean
	sigma(1,c) = cstd;          % store column std
end;

% explore data again
plot(X_norm(:,1), y, 'bx', 'MarkerSize', 8);
print -djpg 'fig2_X_col1_normalized.jpg';

X_norm = [ones(m, 1) X_norm];   % Add intercept term to X_norm


% ====== Using gradient descent ======

% give learning rate and step counts
alpha = 0.3;
step_ct = 50;

theta = zeros(3,1);             % initialize theta to zeros

cost_h = zeros(step_ct,1);      % create tracker to record cost function history

% perform gradient descent
for num = 1:step_ct
	
	temp_var = zeros(size(theta,1),1);                      % temp_var to store new theta
	
    % loop throw each element of theta and store new value in temp_var
	for r=1:size(theta,1)
	    temp_var(r) = theta(r) - alpha * ((1/m) * sum((X_norm*theta-y).*(X_norm(:,r))));
	end;
	
	theta = temp_var;           % assign theta to new theta
	
	cost_h(num) = (X_norm*theta-y)' * (X_norm*theta-y) / (2*m);       % store cost for this step
	
end;

% plot
figure;
plot(1:step_ct, cost_h, '-b', 'LineWidth', 2);              % plot cost value as step increases
xlabel('Number of steps');
ylabel('Cost value');
print -djpg 'fig3_gradient_descent.jpg';

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('The result should be: [305394.99450; -31132.036292; -60241.377398]\n');


% ======= Using normal equation =======

% intuition: take the derivative of cost function and set to 0, but the below function will work!

theta_n = pinv(X_norm' * X_norm) * X_norm' * y;

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta_n);
fprintf('The result should be: [305395.0; -31132.030004; -60241.384184]\n');