# Gradient-enhanced-neuronal-network
In case of computer generated training data, e.g. with FEM or CFD, it is easy to determine the gradients of the output variable according to the problem internal parameters. These additional data can be included in the neural network. For this the loss function is extended by the norm of the gradient observations, i.e:
$$\mathcal{L} = \left \| y-\hat{y} \right \| + \left \| \nabla y-\nabla \hat{y} \right \| $$
where we calculate $$\nabla \hat{y} $$ using tensorflows gradient tape within the loss function.
