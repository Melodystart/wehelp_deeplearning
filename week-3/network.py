class Network:
    def __init__(self, layers, weights, weights_biases, biases):
        self.layers = layers
        self.weights = weights
        self.weights_biases = weights_biases
        self.biases = biases

    def get_weight(self, layer, i, j):
        return self.weights[layer][i][j]
    
    def get_weight_bias(self, layer, j):
        return self.weights_biases[layer][j]

    def get_bias(self, layer):
        return self.biases[layer]

    def get_layer_output(self, layer, inputs):
        layer_weight = self.weights[layer]
        layer_weight_bias = self.weights_biases[layer]     
        layer_output = []
        for i in range(len(layer_weight[0])):
            item = 0
            for j in range(len(inputs)):
                item += inputs[j] * self.get_weight(layer, j, i)
            item += layer_weight_bias[i]*self.get_bias(layer)
            layer_output.append(item)
        return layer_output
        
    def forward(self, inputs):
        for layer in range(self.layers):
            layer_output = self.get_layer_output(layer, inputs)
            if layer == self.layers-1:
                return layer_output
            else:
                inputs = layer_output

print("------ Model 1 ------")
network = Network(2,[[[0.5, 0.6],[0.2, -0.6]],[[0.8],[0.4]]],[[0.3, 0.25],[-0.5]],[1,1])
outputs = network.forward([1.5, 0.5])
print(outputs)
outputs = network.forward([0, 1])
print(outputs)

print("------ Model 2 ------")
network = Network(3,[[[0.5, 0.6],[1.5, -0.8]],[[0.6],[-0.8]],[[0.5, -0.4]]],[[0.3, 1.25],[0.3],[0.2, 0.5]],[1,1,1])
outputs = network.forward([0.75, 1.25])
print(outputs)
outputs = network.forward([-1, 0.5])
print(outputs)

# ------ Model 1 ------
# [0.7599999999999998]
# [-0.24]
# ------ Model 2 ------
# [0.835, -0.007999999999999896]
# [0.41500000000000004, 0.32799999999999996]