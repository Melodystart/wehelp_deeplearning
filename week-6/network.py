import math
import csv
import statistics
import random
import torch

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
        epsilon = 1e-10  # 避免數值為 0 或 1
        return max(min(1 / (1 + math.exp(-x)), 1 - epsilon), epsilon)
        # return 1/(1+math.exp(-x))
    
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
                # print(gradient_bias, output_losses[i], activation_derivative, last_layer_bias)
                layer_gradients_bias.append(gradient_bias)

                # 第layer層的第i個output的第j個weight
                for j in range(len(self.weights[layer][i])):
                    # print(i,j)
                    last_layer_output = self.outputs[layer][j]                 
                    activation_derivative = self.activation_derivative(activations[layer], output_before_activation)
                    gradient = output_losses[i] * activation_derivative * last_layer_output
                    # print(gradient, output_losses[i], activation_derivative, last_layer_output)
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
        return sum([(expect-output)**2 for expect, output in zip(expects, outputs)])/len(expects)
    
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

def random_weight(x, y, z):
  w1 = []
  w2 = []
  b1 = []
  b2 = []
  for i in range(y):
    a = []
    b1.append(random.uniform(-1, 1))
    for j in range(x):
      a.append(random.uniform(-1, 1))
    w1.append(a)

  for k in range(z):
    b = []
    b2.append(random.uniform(-1, 1))
    for i in range(y):
      b.append(random.uniform(-1, 1))
    w2.append(b)
  return w1, w2, b1, b2

print("------ Task 1 ------")
file = open('gender-height-weight.csv')
reader = csv.reader(file)
data_list = list(reader)
file.close()
gender = []
height = []
weight = []

xs = []
es = []

for i in range(1, len(data_list)):
    gender.append(data_list[i][0])
    height.append(float(data_list[i][1]))
    weight.append(float(data_list[i][2]))

height_mean = statistics.mean(height)
height_pstdev = statistics.pstdev(height)

weight_mean = statistics.mean(weight)
weight_pstdev = statistics.pstdev(weight)

for i in range(len(gender)):
    x = []
    if gender[i] == "Male":
        x.append(1)
    else:
        x.append(0)

    x.append((height[i] - height_mean) / height_pstdev)
    xs.append(x)
    es.append((weight[i] - weight_mean) / weight_pstdev)

w1, w2, b1, b2 = random_weight(2, 2, 1)
nn = Network(2,[w1,w2],[b1,b2],[1, 1])
loss_fn = MSE()
learning_rate = 0.01
activations = ["Linear","Linear"]

print("------ Before Training ------")
loss_sum=0 
for x, e in zip(xs, es): 
    expects = [e]
    outputs=nn.forward([x[0], x[1]], activations)
    loss=loss_fn.get_total_loss ( outputs , expects ) 
    loss_sum+=loss 
avg_loss=loss_sum/len(xs)
avg_loss_decode = (avg_loss * ( weight_pstdev ** 2)) ** 0.5
print("Average Loss in Weight", avg_loss_decode)

print("------ Start Training ------")
times = 100
for t in range(times):
    for x, e in zip(xs, es):
        expects = [e]
        outputs = nn.forward([x[0], x[1]], activations)
        loss = loss_fn.get_total_loss(outputs , expects)
        output_losses = loss_fn.get_output_losses()
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)

print("------ Start Evaluating ------")
loss_sum=0 
for x, e in zip(xs, es): 
    expects = [e]
    outputs=nn.forward([x[0], x[1]], activations)
    loss=loss_fn.get_total_loss ( outputs , expects ) 
    loss_sum+=loss 
avg_loss=loss_sum/len(xs)
avg_loss_decode = (avg_loss * ( weight_pstdev ** 2)) ** 0.5
print("Average Loss in Weight", avg_loss_decode)
print("\n")

print("------ Task 2 ------")
file = open('titanic.csv')
reader = csv.reader(file)
data_list = list(reader)
file.close()

survived = []
pclass = []
sex = []
age = []
sibsp = []
parch = []
fare = []
cabin = []
cabin_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
embarked = []

xs = []
es = []

for i in range(1, len(data_list)):
    survived.append(int(data_list[i][1]))
    pclass.append(int(data_list[i][2]))
    sex.append(1 if data_list[i][4] == "male" else 0)
    if data_list[i][5] == None:
        age.append(0)
    elif data_list[i][5] != "":
        age.append(float(data_list[i][5]))
    else:
        age.append(0)
    sibsp.append(int(data_list[i][6]))
    parch.append(int(data_list[i][7]))

    fare.append(float(data_list[i][9]))
    if data_list[i][10] == None or data_list[i][10] == "":
        cabin.append(0)
    else:
        cabin.append(cabin_dict[data_list[i][10][0]])

    if data_list[i][11] == "C":
        embarked.append(0)
    elif data_list[i][11] == "Q":
        embarked.append(1)
    else:
        embarked.append(2)

age_sorted = sorted(age)
age_median = statistics.median(age_sorted)
age = [age_median if a == 0 else float(a) for a in age]

for i in range(len(survived)):
    xs.append([pclass[i], sex[i], age[i], sibsp[i], parch[i], fare[i], cabin[i], embarked[i]])
    es.append(survived[i])

w1, w2, b1, b2 = random_weight(8, 2, 1)
nn = Network(2,[w1,w2],[b1,b2],[1, 1])
loss_fn = BinaryCrossEntropy()
learning_rate = 0.01
activations = ["Linear","Sigmoid"]

print("------ Before Training ------")
correct_count = 0 
threshold = 0.5 
for x, e in zip(xs, es): 
    output = nn.forward([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]], activations)
    survival_status = 0 
    if output[0] > threshold: 
        survival_status = 1 
    if survival_status == e: 
        correct_count += 1 
correct_rate = correct_count/len(xs)
print("Correct Rate", correct_rate*100, "%")

print("------ Start Training ------")
times = 200
for t in range(times):
    for x, e in zip(xs, es):
        expects = [e]
        outputs = nn.forward([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]], activations)
        loss = loss_fn.get_total_loss(outputs , expects)
        output_losses = loss_fn.get_output_losses()
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)

print("------ Start Evaluating ------")
correct_count = 0 
threshold = 0.5 
for x, e in zip(xs, es): 
    output = nn.forward([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]], activations)
    survival_status = 0 
    if output[0] > threshold: 
        survival_status = 1 
    if survival_status == e: 
        correct_count += 1 
correct_rate = correct_count/len(xs)
print("Correct Rate", correct_rate*100, "%")
print("\n")

print("------ Task 3-1 ------")
data = [[2, 3, 1], [5, -2, 1]]
tensor = torch.tensor(data)
print(tensor.dtype, tensor.shape)

print("------ Task 3-2 ------")
shape = (3,4,2)
rand_tensor = torch.rand(shape)
print(rand_tensor.shape)
print(rand_tensor)

print("------ Task 3-3 ------")
shape = (2,1,5)
ones_tensor = torch.ones(shape)
print(ones_tensor.shape)
print(ones_tensor)

print("------ Task 3-4 ------")
mat1 = torch.tensor([[1, 2, 4], [2, 1, 3]])
mat2 = torch.tensor([[5], [2], [1]])
matmul_tensor = torch.matmul(mat1, mat2)
print(matmul_tensor.shape)
print(matmul_tensor)

print("------ Task 3-5 ------")
mat1 = torch.tensor([[1, 2], [2, 3], [-1, 3]])
mat2 = torch.tensor([[5, 4], [2, 1], [1, -5]])
mul_tensor = mat1*mat2
print(mul_tensor.shape)
print(mul_tensor)


# ------ Task 1 ------
# ------ Before Training ------
# Average Loss in Weight 75.58530107058495
# ------ Start Training ------
# ------ Start Evaluating ------
# Average Loss in Weight 10.995985084979797


# ------ Task 2 ------
# ------ Before Training ------
# Correct Rate 62.62626262626263 %
# ------ Start Training ------
# ------ Start Evaluating ------
# Correct Rate 78.33894500561168 %


# ------ Task 3-1 ------
# torch.int64 torch.Size([2, 3])
# ------ Task 3-2 ------
# torch.Size([3, 4, 2])
# tensor([[[0.4011, 0.8773],
#          [0.1668, 0.5485],
#          [0.4610, 0.4371],
#          [0.8604, 0.3102]],

#         [[0.1982, 0.5274],
#          [0.9294, 0.4482],
#          [0.8416, 0.7458],
#          [0.8786, 0.5670]],

#         [[0.3151, 0.6946],
#          [0.6130, 0.3272],
#          [0.4298, 0.3383],
#          [0.7319, 0.6923]]])
# ------ Task 3-3 ------
# torch.Size([2, 1, 5])
# tensor([[[1., 1., 1., 1., 1.]],

#         [[1., 1., 1., 1., 1.]]])
# ------ Task 3-4 ------
# torch.Size([2, 1])
# tensor([[13],
#         [15]])
# ------ Task 3-5 ------
# torch.Size([3, 2])
# tensor([[  5,   8],
#         [  4,   3],
#         [ -1, -15]])