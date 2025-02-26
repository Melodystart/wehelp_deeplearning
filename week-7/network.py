from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch 
import csv
import statistics

class MyData(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = label.clone().detach().float()

    def __getitem__(self, index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.label)

print("------ Task 1 ------")
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2,2),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    test_loss_decode = (test_loss * ( weight_pstdev ** 2)) ** 0.5
    return test_loss_decode

# Prepare Data with Dataset
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

print("------ Before Training ------")
es = torch.tensor(es, dtype=torch.float32).view(-1, 1)
dataset = MyData(xs, es)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

avg_loss = test(test_loader, model, loss_fn)
print("Average Loss in Weight", avg_loss)

print("------ Start Training ------")
epochs = 30
for t in range(epochs):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    avg_loss = test(test_loader, model, loss_fn)

print("Average Loss in Weight", avg_loss)


print("------ Task 2 ------")
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(8,4),
            nn.Linear(4,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).round()
            correct += (pred.eq(y).sum().item())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    # print(f"Loss: {test_loss}")
    correct_rate = correct / size
    return correct_rate

# Prepare Data with Dataset
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

print("------ Before Training ------")
es = torch.tensor(es, dtype=torch.float32).view(-1, 1)
dataset = MyData(xs, es)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

correct_rate = test(test_loader, model, loss_fn)
print("Correct Rate", correct_rate*100, "%")

print("------ Start Training ------")
epochs = 300
for t in range(epochs):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    correct_rate = test(test_loader, model, loss_fn)

print("Correct Rate", correct_rate*100, "%")


# ------ Task 1 ------
# ------ Before Training ------
# Average Loss in Weight 40.00715944909176
# ------ Start Training ------
# Average Loss in Weight 10.229882210085558

# ------ Task 2 ------
# ------ Before Training ------
# Correct Rate 30.726256983240223 %
# ------ Start Training ------
# Correct Rate 79.3296089385475 %