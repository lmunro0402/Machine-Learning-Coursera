function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 

rtheta = theta(2:end); 
h = sigmoid(X*theta);


J = 1/m * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + lambda/(2*m) * sum(rtheta.^2);

% grad(1) is multiple values hmm... jk it's computing one vector of probability at a time not a matrix of all possibilities

%grad(2:end) = (1 / m) * (X(:,2:end)' * (h - y) ) + lambda/m * rtheta;


%a = (1 / m) * ((h - y)' * X(:,2:end)) + (lambda/m * rtheta)';
%size(a)
%grad(2:end) = a';

% ^^^^ gross ^^^^ leave here for understanding later. Really only hard part here is formatting the matrices
% a tool to see matrices dimensions on the side would be very helpful

% when you look back at this think about why grad(1) can be h-y' * X(:,1) but grad(2:end) must be the other way around 
% reason: 2 x 1 vs a 1 x 2
% note: a'b = b'a is true for SUMS but not if the result is a matrix of greater that 1 x 1

%grad(1) = (1 / m) * ((h - y)' * X(:,1))

%size((h-y)' * X)
%size(X'*(h-y))
%size(theta)
grad = ((1 / m) * (X' * (h - y)) + lambda/m * [0; rtheta]);
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

%grad = grad(:);

end
