import math

class Network:
    def __init__(self, layers, weights, weights_biases, biases):
        self.layers = layers
        self.weights = weights
        self.weights_biases = weights_biases
        self.biases = biases

    def get_weight(self, layer, j, i):
        return self.weights[layer][j][i]
    
    def get_weight_bias(self, layer, i):
        return self.weights_biases[layer][i]

    def get_bias(self, layer):
        return self.biases[layer]

    def get_layer_output(self, layer, inputs):
        layer_weight = self.weights[layer]    
        layer_output = []
        for i in range(len(layer_weight[0])):
            item = 0
            for j in range(len(inputs)):
                item += inputs[j] * self.get_weight(layer, j, i)
            item += self.get_bias(layer) * self.get_weight_bias(layer, i)
            layer_output.append(item)
        return layer_output

    def Linear(self, x):
        return x
    
    def ReLu(self, x):
        if x > 0:
            return x
        else:
            return 0
        
    def Sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
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
        
    def forward(self, inputs, activation1, activation2):
        for layer in range(self.layers):
            layer_outputs = self.get_layer_output(layer, inputs)
            if layer == self.layers-1:
                return self.activation(activation2, layer_outputs)
            else:
                inputs = self.activation(activation1, layer_outputs)

def MSE(self, ouputs, expects):
    return sum([(expect-ouput)**2 for expect, ouput in zip(expects, ouputs)])/len(expects)

def BinaryCrossEntropy(self, outputs, expects):
    return -sum([expect*math.log(ouput)+(1-expect)*math.log(1-ouput) for expect, ouput in zip(expects, outputs)])

def CategoricalCrossEntropy(self, outputs, expects):
    return -sum([expect*math.log(ouput) for expect, ouput in zip(expects, outputs)])

print("------ Model 1 ------")
network = Network(2,[[[0.5, 0.6],[0.2, -0.6]],[[0.8,0.4],[-0.5,0.5]]],[[0.3, 0.25],[0.6,-0.25]],[1,1])

outputs = network.forward([1.5, 0.5], "ReLu", "Linear")
print("Outputs", outputs)
print("Total Loss", MSE(network, outputs, [0.8, 1]))

outputs = network.forward([0, 1], "ReLu", "Linear")
print("Outputs", outputs)
print("Total Loss", MSE(network, outputs, [0.5, 0.5]))

print("------ Model 2 ------")
network = Network(2,[[[0.5, 0.6],[0.2, -0.6]],[[0.8],[0.4]]],[[0.3, 0.25],[-0.5]],[1,1])
outputs = network.forward([0.75, 1.25], "ReLu", "Sigmoid")
print("Outputs", outputs)
print("Total Loss", BinaryCrossEntropy(network, outputs, [1]))

outputs = network.forward([-1, 0.5], "ReLu", "Sigmoid")
print("Outputs", outputs)
print("Total Loss", BinaryCrossEntropy(network, outputs, [0]))

print("------ Model 3 ------")
network = Network(2,[[[0.5, 0.6],[0.2, -0.6]],[[0.8, 0.5, 0.3],[-0.4, 0.4, 0.75]]],[[0.3, 0.25],[0.6, 0.5, -0.5]],[1,1])
outputs = network.forward([1.5, 0.5], "ReLu", "Sigmoid")
print("Outputs", outputs)
print("Total Loss", BinaryCrossEntropy(network, outputs, [1, 0, 1]))

outputs = network.forward([0, 1], "ReLu", "Sigmoid")
print("Outputs", outputs)
print("Total Loss", BinaryCrossEntropy(network, outputs, [1, 1, 0]))

print("------ Model 4 ------")
network = Network(2,[[[0.5, 0.6],[0.2, -0.6]],[[0.8, 0.5, 0.3],[-0.4, 0.4, 0.75]]],[[0.3, 0.25],[0.6, 0.5, -0.5]],[1,1])
outputs = network.forward([1.5, 0.5], "ReLu", "Softmax")
print("Outputs", outputs)
print("Total Loss", CategoricalCrossEntropy(network, outputs, [1, 0, 0]))

outputs = network.forward([0, 1], "ReLu", "Softmax")
print("Outputs", outputs)
print("Total Loss", CategoricalCrossEntropy(network, outputs, [0, 0, 1]))

# ------ Model 1 ------
# Outputs [1.095, 0.6349999999999999]
# Total Loss 0.11012500000000001
# Outputs [1.0, -0.04999999999999999]
# Total Loss 0.27625
# ------ Model 2 ------
# Outputs [0.5597136492671929]
# Total Loss 0.5803299666264259
# Outputs [0.3775406687981454]
# Total Loss 0.47407698418010663
# ------ Model 3 ------
# Outputs [0.7649478037637647, 0.8045533772735622, 0.6183380393302829]
# Total Loss 2.381135626086851
# Outputs [0.7310585786300049, 0.679178699175393, 0.41338242108267]
# Total Loss 1.2335148490519
# ------ Model 4 ------
# Outputs [0.3619598853670419, 0.45784623293032367, 0.18019388170263428]
# Total Loss 1.0162218871998794
# Outputs [0.49066725279292916, 0.3821320407026282, 0.12720070650444248]
# Total Loss 2.0619890738113322