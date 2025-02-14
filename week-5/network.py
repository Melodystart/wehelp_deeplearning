import math

class Network:
    def __init__(self, layers, weights, weights_biases, biases):
        self.layers = layers
        self.weights = weights
        self.weights_biases = weights_biases
        self.biases = biases
        self.outputs_before_activation = []
        self.outputs = []
        self.output_losses = {}
        self.gradients = []
        self.gradients_bias = []

    def get_weight(self, layer, i, j):
        return self.weights[layer][i][j]
    
    def get_weight_bias(self, layer, i):
        return self.weights_biases[layer][i]

    def get_bias(self, layer):
        return self.biases[layer]

    def get_layer_output(self, layer, inputs):
        layer_weight = self.weights[layer]    
        layer_output = []
        for i in range(len(layer_weight)):
            item = 0
            for j in range(len(inputs)):
                item += inputs[j] * layer_weight[i][j]
            item += self.get_bias(layer) * self.get_weight_bias(layer, i)
            layer_output.append(item)
        return layer_output

    def Linear(self, x):
        return x
    
    def Linear_Derivative(self, x):
        return 1
    
    def ReLu(self, x):
        if x > 0:
            return x
        else:
            return 0
    
    def ReLu_Derivative(self, x):
        if x > 0:
            return 1
        else:
            return 0
        
    def Sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def Sigmoid_Derivative(self, x):
        return self.Sigmoid(x)*(1-self.Sigmoid(x))
    
    def Softmax(self, xs):
        max_x = max(xs)
        exp_xs = [math.exp(x-max_x) for x in xs]
        sum_exp_xs = sum(exp_xs)
        return [exp_x/sum_exp_xs for exp_x in exp_xs]

    def activation(self, activation, layer_outputs):
        if activation == "Linear":
            return [self.Linear(x) for x in layer_outputs]
        elif activation == "ReLu":
            return [self.ReLu(x) for x in layer_outputs]
        elif activation == "Sigmoid":
            return [self.Sigmoid(x) for x in layer_outputs]
        elif activation == "Softmax":
            return self.Softmax(layer_outputs)
    
    def activation_derivative(self, activation, x):
        if activation == "Linear":
            return self.Linear_Derivative(x)
        elif activation == "ReLu":
            return self.ReLu_Derivative(x)
        elif activation == "Sigmoid":
            return self.Sigmoid_Derivative(x)

    def forward(self, inputs, activations):
        self.outputs = []
        self.outputs_before_activation = []
        self.outputs_before_activation.append(inputs)
        for layer in range(self.layers):
            self.outputs.append(inputs)
            layer_outputs = self.get_layer_output(layer, inputs)
            if layer == self.layers-1:
                self.outputs_before_activation.append(layer_outputs)
                results  = self.activation(activations[layer], layer_outputs)
                self.outputs.append(results)
                return results
            else:
                self.outputs_before_activation.append(layer_outputs)
                inputs = self.activation(activations[layer], layer_outputs)
    
    def backward(self, output_losses):
        self.gradients = []
        self.gradients_bias = []
        for layer in range(self.layers-1, -1, -1):
            layer_gradients = []
            layer_gradients_bias = []
            if layer < self.layers-1:
                output_losses = list(self.output_losses[layer+1].values())
            self.output_losses[layer] = {}
            # 第layer層的第i個output
            for i in range(len(self.weights[layer])):
                layer_output_gradients = []
                output_before_activation = self.outputs_before_activation[layer+1][i]
                last_layer_bias = self.biases[layer]   
                activation_derivative = self.activation_derivative(activations[layer], output_before_activation)
                gradient_bias = output_losses[i] * activation_derivative * last_layer_bias
                layer_gradients_bias.append(gradient_bias)

                # 第layer層的第i個output的第j個weight
                for j in range(len(self.weights[layer][i])):
                    last_layer_output = self.outputs[layer][j]                 
                    activation_derivative = self.activation_derivative(activations[layer], output_before_activation)
                    gradient = output_losses[i] * activation_derivative * last_layer_output
                    if j not in self.output_losses[layer]:
                        self.output_losses[layer][j] = self.weights[layer][i][j] * output_losses[i] * activation_derivative
                    else:
                        self.output_losses[layer][j] += self.weights[layer][i][j] * output_losses[i] * activation_derivative
                    layer_output_gradients.append(gradient)
                layer_gradients.append(layer_output_gradients)
            self.gradients.insert(0, layer_gradients)
            self.gradients_bias.insert(0, layer_gradients_bias)

    def zero_grad(self, learning_rate):
        for layer in range(self.layers):
            for i in range(len(self.weights[layer])):
                self.weights_biases[layer][i] -= learning_rate * self.gradients_bias[layer][i]
                for j in range(len(self.weights[layer][i])):
                    self.weights[layer][i][j] -= learning_rate * self.gradients[layer][i][j]

    def print_weights(self):
        for layer in range(self.layers):
            print("Layer", layer)
            print(self.weights[layer])
            print(self.weights_biases[layer])

class MSE:
    def __init__(self):
        self.outputs = []
        self.expects = []

    def get_total_loss(self, outputs , expects):
        self.outputs = outputs
        self.expects = expects
        return sum([(expect-ouput)**2 for expect, ouput in zip(expects, outputs)])/len(expects)
    
    def get_output_losses(self):
        return [(ouput-expect)*2/len(self.expects) for expect, ouput in zip(self.expects, self.outputs)]
    
class BinaryCrossEntropy:
    def __init__(self):
        self.outputs = []
        self.expects = []

    def get_total_loss(self, outputs , expects):
        self.outputs = outputs
        self.expects = expects
        return -sum([expect*math.log(output)+(1-expect)*math.log(1-output) for expect, output in zip(expects, outputs)])

    def get_output_losses(self):
        return [(-expect/output + (1-expect)/(1-output)) for expect, output in zip(self.expects, self.outputs)]


def CategoricalCrossEntropy(self, outputs, expects):
    return -sum([expect*math.log(ouput) for expect, ouput in zip(expects, outputs)])

print("------ Model 1 ------")
nn = Network(3,[[[0.5, 0.2],[0.6, -0.6]],[[0.8, -0.5]],[[0.6], [-0.3]]],[[0.3, 0.25],[0.6],[0.4, 0.75]],[1, 1, 1])
expects = [0.8, 1]
loss_fn = MSE()
learning_rate = 0.01
activations = ["ReLu", "Linear", "Linear"]

print("-----Task 1-----")
for i in range(1000):
    outputs = nn.forward([1.5, 0.5], activations)
    loss = loss_fn.get_total_loss(outputs , expects)
    output_losses = loss_fn.get_output_losses()
    nn.backward(output_losses)
    nn.zero_grad(learning_rate)

    if i == 0:
        nn.print_weights()
print("-----Task 2-----")        
print("Total Loss", loss_fn.get_total_loss(outputs, expects))


print("------ Model 2 ------")
nn = Network(2,[[[0.5, 0.2],[0.6, -0.6]],[[0.8, 0.4]]],[[0.3, 0.25],[-0.5]],[1, 1])
expects = [1]
loss_fn = BinaryCrossEntropy()
learning_rate = 0.1
activations = ["ReLu", "Sigmoid"]

print("-----Task 1-----")
for i in range(1000):
    outputs = nn.forward([0.75, 1.25], activations)
    loss = loss_fn.get_total_loss(outputs , expects)
    output_losses = loss_fn.get_output_losses()
    nn.backward(output_losses)
    nn.zero_grad(learning_rate)

    if i == 0:
        nn.print_weights()
print("-----Task 2-----")        
print("Total Loss", loss_fn.get_total_loss(outputs, expects))



# ------ Model 1 ------
# -----Task 1-----
# Layer 0
# [[0.496067, 0.198689], [0.602458125, -0.599180625]]
# [0.297378, 0.25163875]
# Layer 1
# [[0.796230875, -0.502785875]]
# [0.5967224999999999]
# Layer 2
# [[0.59718585], [-0.293665425]]
# [0.39743, 0.755785]
# -----Task 2-----
# Total Loss 3.913889781826326e-15


# ------ Model 2 ------
# -----Task 1-----
# Layer 0
# [[0.5264171810439684, 0.2440286350732807], [0.6, -0.6]]
# [0.33522290805862454, 0.25]
# Layer 1
# [[0.8407264874427847, 0.4]]
# [-0.4559713649267193]
# -----Task 2-----
# Total Loss 0.00041967159860447034