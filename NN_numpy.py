import numpy as np

# dense layer
class Dense:
    def __init__(self, n_in, n_neurons):
        self.weights = 0.01 * np.random.randn(n_in , n_neurons)   # w^T shape
        self.baises = np.zeros((1, n_neurons))                   # b^T shape

    def __call__(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises
        return self.output
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.baises
        return self.output
         
    def backward(self, prev_grads):
        self.dweights = np.dot(self.inputs.T, prev_grads)       # inputs.T = (no. of features, no. of samples)
        self.dbiases = np.sum(prev_grads, axis=0, keepdims=True)    # sum over columns (i.e wrt to each neuron)
        # gradients on inputs
        # each dinputi will be sum over dai*wi for all neurons, da is upstream gradient
        # upstream grad shape = (m , self.n_neurons) 
        # self.weights.T shape = (self.n_neurons, self.n_in)
        # dinputs shape = (m, self.n_in)    [n_in are the n_neurons for prev layer]
        self.dinputs = np.dot(prev_grads, self.weights.T)   # required to propogate gradients

    def __repr__(self):
        return f"Dense Layer ({self.weights.shape[0]} -> {self.weights.shape[1]})"
    
    def __str__(self):
        return self.__repr__()

# activation functions
class ReLU:
    def __call__(self, inputs ):
        self.output = np.maximum(0, inputs)
        return self.output
    
    def forward(self, inputs ):
        # relu = 0 for x < 0
        #      = x for x >= 0
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, prev_grads):
        #prev_grads shape = (m , self.n_neurons)
        #self.outputs shape = (m , self.n_neurons)
        self.dinputs = prev_grads.copy()
        # dinputs = self.output * prev_grad         ---- element wise multiplication not dot product
        self.dinputs[prev_grads <= 0] = 0 

class Sigmoid:
    def __call__(self, inputs):
        exp = np.exp(inputs)
        self.output = exp/(exp + 1)
        return self.output

    def forward(self, inputs):
        # sigmoid(x) = 1/1+e^-x = e^x / 1 + e^x
        exp = np.exp(inputs)
        self.output = exp/(exp + 1)
        return self.output
    
    def backward(self, prev_grads):
        # sigmoid(x) = e^x / 1 + e^x
        # sigmoid'(x) = e^x(1 + e^x) * 1/1+e^x
        #             = sigmoid(x) * (1 - sigmoid(x))

        # prev_grads shape = (m , self.n_neurons)
        # self.output shape = (m , self.n_neurons)
        self.dinputs = (self.output*(1-self.output)) * prev_grads   # element-wise multiplication

class Softmax:
    def __call__(self, inputs):
        # sofmax(xi) = e^-
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # to prevent exploding gradients by preventing overflow from exponent function
        norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output
    
    def forward(self, inputs):
        # sofmax(xi) = e^-xi / sum(e^xj) j = 1 to n_neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # to prevent exploding gradients by preventing overflow from exponent function
        norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values
        return self.output
    
    def backward(self, prev_grads):
        # sofmax(xi) = e^-xi / sum(e^xj) j = 1 to n_neurons
        # dsoftmax(xi)/dxi = softmax(xi)*(1-sofmax(xi))
        # dsoftmax(xi)/dxj = -softmax(xi)*softmax(xj)

        # dsoftmax(xi)dxj = softmax(xi)*(1-sofmax(xi))   i = j
        #                 = -softmax(xi)*softmax(xj)    i != j
        # Kronecker delta function dij = 1 if i = j 
        #                              = 0 i!=j
        # dsoftmax(xi)/dxj = softmax(xi)(dij - softmax(xj))
        #                  = softmax(xi)dij - softmax(xi)softmax(xj)

        # but xi = ipi - max
        # dxi = 1

        # prev_grads shape = (m , n_classes)
        # self.output shape = (m, n_classes)
        self.dinputs = np.empty_like(prev_grads)

        # enumerate outputs and gradients
        for index, (single_output, single_grad) in enumerate(zip(self.output, prev_grads)):
            # flatten output array
            single_output = single_output.reshape(-1,1) # col matrix
            # Jacobian matrix is an array of partial derivatives in all of the combinations of both input vectors.
            # Jacobian matrix of a vector-valued function of several variables is the matrix of all its first-order partial derivatives
            # calculate jacobian matrix of the output s
            # comes from jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # calculate sample-wise gradient
            # and add it to the array of sample gradients
            # here single grad is shape (1, )
            # numpy treats it as column vector when it is on the left of dot product
            self.dinputs[index] = np.dot(jacobian_matrix, single_grad)  # shape(n_classes, 1) but numpy gives shape (1, ) which will be row vector in out context



# loss functions
class Loss:

    # calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # to prevent division by zero(log(0))   
        clipped_output = np.clip(y_pred, 1e-07, 1-1e-07)    # 1-1e-07 becuase if there is some epsilon addition the log(x > 1) will be negative instead of 0

        # Probablities for target values -
        # only if categorical labels    i.e true class index for each sample
        if len(y_true.shape) == 1:
            correct_confidences = clipped_output[range(samples), y_true]    # y_true only contains the incide of true class
        elif len(y_true.shape) ==2:     # contains one-hot encoded labels OR true probability distribution
            # mask the valus
            correct_confidences = np.sum(clipped_output*y_true, axis=1)
        
        #losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)       # number of samples m
        labels = len(y_pred[0])     # number of classes c

        # prev_grad shape = (m, c)

        # if labels are sparse convert to one hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]         #[y_true] reorders the values of I to match the hot encoding
        
        self.dinputs = -y_true / y_pred

        # normalization of gradients
        self.dinputs = self.dinputs/samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
# no jacobian calculation involved
class Activation_Softmax_Loss_Cross_Entropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CrossEntropyLoss()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.forward(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        """Derivation of combined Softmax + Cross-Entropy gradient:

        1. Softmax converts logits z_j into probabilities p_j.
            p_j = e^(z_j) / sum(e^(z_k))

        2. Cross-entropy loss for one-hot labels:
            L = -log(p_c)      (c = correct class)

        3. We want dL/dz_j (gradient of loss wrt logits).

        4. Use the chain rule:
            dL/dz_j = sum_i( dL/dp_i * dp_i/dz_j )

        5. Derivative of cross-entropy wrt softmax output:
            dL/dp_i = -y_i / p_i
            (non-zero only for the correct class because y is one-hot)

        6. Derivative of softmax wrt logits:
            if i == j: dp_i/dz_j = p_i * (1 - p_i)
            else:      dp_i/dz_j = -p_i * p_j

        7. Multiply them using the chain rule: the complicated terms cancel out.

        8. After simplification the entire derivative collapses to:
            dL/dz_j = p_j - y_j

        9. Meaning:
            - correct class:  p_j - 1
            - other classes:  p_j - 0

        10. This is why the combined backward pass can simply do:
                dinputs = softmax_output
                dinputs[range(samples), true_class] -= 1
                dinputs /= samples
            which implements: (p - y) / batch_size  """

        # in the combined backward pass:
        # dvalues = softmax output (p_j)  -> NOT upstream gradients
        # because the loss layer starts backprop, so it receives the model's predictions
        
        samples = len(dvalues)  # number of samples m
        # dvalues shape = (m, n_classes)

        # if true labels are one-hot encoded convert them to sparse labels
        # because we only need the index where y_j = 1
        # and subtracting 1 at that index will give (p_j - y_j)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)  
            # y_true now shape = (m, ), each entry is correct class index

        # make a copy so we don't modify original softmax output
        # starting gradient = p_j
        # final gradient = p_j - y_j
        self.dinputs = dvalues.copy()

        # subtract 1 from the softmax output at the correct class index
        # this performs: p_j - 1  for the correct class
        #                p_j - 0  for other classes
        # since y_j (one-hot) = 1 only at the correct class
        # this exactly implements the derived formula:
        #   dL/dz_j = p_j - y_j
        self.dinputs[range(samples), y_true] -= 1

        # normalization
        self.dinputs = self.dinputs / samples

# optimizers
class Optimizer_SGD:
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# regularization
class Dropout:
    pass

class BatchNorm:
    pass
