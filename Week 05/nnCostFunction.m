function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
a1 = [ones(m, 1) X];

a2 = sigmoid(a1 * Theta1');
a2 = [ones(m,1) a2];

a3 = sigmoid(a2 * Theta2');

for i=1:m
    y_vec = zeros(num_labels,1);
    index = y(i);
    y_vec(index) = 1; 
    
    J = J + (log(a3(i,:)) * y_vec + log(1-a3(i,:)) * (1 - y_vec));
end

Theta1_reg = Theta1;
Theta1_reg(:,1) = [];
 
Theta2_reg = Theta2;
Theta2_reg(:,1) = [];

reg_term = (lambda/(2*m)) * ( sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)) );

J = -(1/m) * J + reg_term;   % end of calculating cost function

% calculating Theta_grad

Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for i=1:m
    y_vec = zeros(num_labels,1);
    index = y(i);
    y_vec(index) = 1;
    
    delta3 = (a3(i,:))' - y_vec;
    
    delta2 = Theta2' * delta3 .* (a2(i,:))' .* (1 - a2(i,:))';
    delta2 = delta2(2:end);
    
    Delta2 = Delta2 + (delta3 * a2(i,:));
    
    Delta1 = Delta1 + (delta2 * a1(i,:));
end

if lambda == 0
  D2 = (1/m) * Delta2;
  Theta2_grad = D2;

  D1 = (1/m) * Delta1;
  Theta1_grad = D1;
else
  D20 = (1/m) * Delta2(:,1); % for j = 0
  D2  = (1/m) * Delta2(:,2:end); % for j >= 1
  Theta2_grad = [D20 , D2 + (lambda/m)*Theta2_reg];
  
  D10 = (1/m) * Delta1(:,1); % for j = 0
  D1  = (1/m) * Delta1(:,2:end); % for j >= 1
  Theta1_grad = [D10 , D1 + (lambda/m)*Theta1_reg];
end










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
