function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h = sigmoid(X * theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta_reg = theta;
theta_reg(1) = [];
X_reg = X;
X_reg(:,1) = [];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = -(1/m) * ( log(h)'* y + log(1-h)' * (1 - y) ) + (lambda/(2*m)) * (theta_reg' * theta_reg);

grad_0 = (1/m)*( sum(h-y) );
grad =   (1/m)*(X_reg' * (h-y)) + (lambda/m) * theta_reg;
grad = [grad_0; grad];
% =============================================================

end
