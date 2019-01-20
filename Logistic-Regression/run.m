% Run file for an example logistic regression model with regularization
% programmer: the one and only :D
%
% data: data1.csv is a 96x3 matrix containing hourly weather at my zipcode for the past 4 days
%       column 1 is temperature in F, column 2 is dew point in F, column 3 indicate clear(0) or overcast/rain(1)
%
% goal: use logistic regression model with regularization to find decision boundary that best fit the given data
%
% regularization: prevent overfitting

% setup and house keeping
clear; close all; clc

% load data
data = load('data1.csv');
X = data(:,1:2);
y = data(:,3);
[m, n] = size(X);            % get input rows and columns

% ====== initial plot ======
figure; hold on;             % create new figure, add hold on to plot two times

% create mask for each layer using labeled(y) list
clear = find(y==0);
rain = find(y==1);

% plot X1 and X2 by mask, identify 2 layers by different marker
plot(X(clear,1), X(clear,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(rain,1), X(rain,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% add label and legend to figure 1
xlabel('Temperature (F)'); 
ylabel('Dew Point (F)');
legend_list = {["Clear"], ["Overcast/Rain"]};
legend(legend_list);
axis([40, 60, 10, 60]);

hold off;

print -djpg 'fig1_explore_data.jpg';                % save figure 1;


% create sigmoid function that computes sigmoid of z and returns g
function g = sigmoid(z)
    g = zeros(size(z));
    [a, b] = size(z);
    for i=1:a,
        for j=1:b,
            g(i,j)=1/(1+e^(-z(i,j)));
        end;
    end;
end

% create cost function to compute cost and gradient
function [J, grad] = costFunction(theta, X, y)
    m = length(y);           % get number of samples
    % compute cost J
    J = (-1/m)*(y'*log(sigmoid(X*theta)) + (1.-y)'*log(1.-sigmoid(X*theta)));

    % compute gradient
    grad = (1/m).*(X'*(sigmoid(X*theta)-y));
end

% ======== computer cost and gradient =========
% in summary, the sigmoid function predits the probability of 1 if true y is 1,
% probability of 0 if true y is 0. To accomplish this, we used a two step log function

X = [ones(m,1) X];            % add intercept term to X

theta1 = zeros(n+1, 1);       % initialize theta

[J, grad] = costFunction(theta1, X, y);              % compute cost and gradient

fprintf('Cost at initial theta (zeros): %f\n', J);
fprintf('Expected cost (approx): 0.693147\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n -0.17708\n -9.06250\n -8.26562\n');


% ======= use fminunc to look for minimum cost ======
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta2, J2] = fminunc(@(t)(costFunction(t, X, y)), theta1, options);


% ======== compute accuracy ========
p_train = round(sigmoid(X*theta2));                  % get prediction from our theta2

fprintf('Train Accuracy: %f\n', mean(double(p_train == y)) * 100);
fprintf('Expected accuracy (approx): 85.4\n');


% ======= plot data with decision boundary =======
figure; hold on;             % create new figure, add hold on to plot two times

% create mask for each layer using labeled(y) list
clear = find(y==0);
rain = find(y==1);

% plot X1 and X2 by mask, identify 2 layers by different marker
plot(X(clear,2), X(clear,3), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(rain,2), X(rain,3), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

line_x_pt = [min(X(:,2))-2,  max(X(:,2))+2];                            % get x values of 2 points
line_y_pt = (-1./theta2(3)).*(theta2(2).*line_x_pt + theta2(1));        % get y values of 2 points
plot(line_x_pt, line_y_pt);                                             % plot the line

% add label and legend to figure 1
xlabel('Temperature (F)'); 
ylabel('Dew Point (F)');
legend_list = {legend_list{:}, ["Decision Boundary"]};
legend(legend_list);
axis([40, 60, 10, 60]);

print -djpg 'fig2_decision_boundary.jpg';             % save figure 2;


% ======= adding polynomial features =======
X_poly = [X X(:,2).^2 X(:,3).^2 X(:,2).*X(:,3)];      % add X1^2, X2^2, X1*X2 polynomial terms  

theta3 = zeros(size(X_poly, 2),1);                    % initialize new theta

lambda = 1;                 % set regularization parameter


% create cost function to compute cost and gradient with regularization
function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);           % get number of samples
    % compute cost J
    J = (-1/m)*(y'*log(sigmoid(X*theta)) + (1.-y)'*log(1.-sigmoid(X*theta))) + lambda/(2*m)*sum(theta(2:end,1).^2);

    % compute gradient
    grad = ((1/m).*(X'*(sigmoid(X*theta)-y))).+([0;theta(2:end,1)].*(lambda/m));
end


% ======= regularization of theta to help reduce overfitting ======
% test: [J3, grad3] = costFunctionReg(theta3, X_poly, y, lambda);

% use fminunc to find optimal theta4
options2 = optimset('GradObj', 'on', 'MaxIter', 400);
[theta4, J4, exit_flag] = fminunc(@(t)(costFunctionReg(t, X_poly, y, lambda)), theta3, options2);


% ======= plot new boundary on existing ======
    % Here is the grid range
    u = linspace(40, 60, 50);
    v = linspace(10, 60, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta4*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = [1 u(i) v(j) u(i).^2 v(j).^2 u(i).*v(j)]*theta4;
        end
    end
    z = z';                    % important to transpose z before calling contour

    % Plot z = 0, Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2);       

legend_list = {legend_list{:}, ["Polynomial Boundary"]};   % not sure why contour plot doesn't generate a legend
legend(legend_list);                                       % investigate later

print -djpg 'fig3_polynomial_boundary.jpg';                % save figure 3;

hold off;

% ======== compute accuracy ========
p_train2 = round(sigmoid(X_poly*theta4));                  % get prediction from our theta4

fprintf('Train Accuracy: %f\n', mean(double(p_train2 == y)) * 100);
fprintf('Expected accuracy (approx): 86.458333\n');