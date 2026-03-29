import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# Function for loading notMNIST Dataset
def loadData(datafile = "notMNIST.npz"):
    with np.load(datafile) as data:
        Data, Target = data["images"].astype(np.float32), data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Custom Dataset class.
class notMNIST(Dataset):
    def __init__(self, annotations, images, transform=None, target_transform=None):
        self.img_labels = annotations
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Define CNN
class CNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(CNN, self).__init__()
        #TODO
        #DEFINE YOUR LAYERS HERE

        # first convolution block:
        # input channels = 1, output channels = 32, kernel size = 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # second convolution block:
        # input channels = 32, output channels = 64, kernel size = 4
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # dropout layer applied after flattening
        self.dropout = nn.Dropout(p=drop_out_p)

        # fully connected layer:
        # flattened feature size = 64 x 4 x 4 = 1024
        # output = 784 hidden units
        self.fc1 = nn.Linear(1024, 784)

        # output layer:
        # input = 784 hidden units
        # output = 10 class logits
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        #TODO
        #DEFINE YOUR FORWARD FUNCTION HERE

        # first convolution block:
        # Conv -> ReLU -> BatchNorm -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)

        # second convolution block:
        # Conv -> ReLU -> BatchNorm -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)

        # flatten feature maps into a 1D vector for each image
        x = torch.flatten(x, start_dim=1)

        # apply dropout
        x = self.dropout(x)

        # hidden fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # final output logits for the 10 classes
        out = self.fc2(x)

        return out

#Define FNN
class FNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(FNN, self).__init__()
        #TODO
        #DEFINE YOUR LAYERS HERE

        # first fully connected layer:
        # input = 28x28 image flattened to 784 features
        # output = 10 hidden units
        self.fc1 = nn.Linear(28 * 28, 10)
        
        # second fully connected layer:
        # input = 10 hidden units
        # output = 10 hidden units
        self.fc2 = nn.Linear(10, 10)

        # dropout layer
        self.dropout = nn.Dropout(p=drop_out_p)

        # output layer:
        # input = 10 hidden units
        # output = 10 class logits
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        #TODO
        #DEFINE YOUR FORWARD FUNCTION HERE

        # flatten each image from (BATCH_SIZE, 1, 28, 28) to (BATCH_SIZE, 784)
        x = torch.flatten(x, start_dim=1)

        # pass through the first hidden layer and apply ReLU activation
        x = F.relu(self.fc1(x))

        # pass through the second hidden layer and apply ReLU activation
        x = F.relu(self.fc2(x))

        # Apply dropout before the final output layer
        x = self.dropout(x)

        # Compute the output logits for the 10 classes
        out = self.fc3(x)

        return out

# Commented out IPython magic to ensure Python compatibility.
# Compute accuracy
def get_accuracy(model, dataloader):

    model.eval()
    device = next(model.parameters()).device
    accuracy = 0.0
    total = 0.0
    correct = 0.0 # add this to find number of correct predictions

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # TODO
            # Return the accuracy

            # forward pass
            outputs = model(images)

            # get the predicted class for each image
            _, predicted = torch.max(outputs, dim=1) # only need the index/class

            # find total samples and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # find sum and convert tensor to int/float
    
    # return accuracy as percentage
    accuracy = (correct / total) * 100
    return accuracy

def train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=50, verbose=False):
  #TODO
  # Define your cross entropy loss function here
  # Use cross entropy loss
  criterion = nn.CrossEntropyLoss()

  #TODO
  # Define your optimizer here
  # Use AdamW optimizer, set the weights, learning rate and weight decay argument.
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  acc_hist = {'train':[], 'val':[], 'test': []}

  for epoch in range(num_epochs):
    model = model.train()
    ## training step
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # TODO
        # Follow the step in the tutorial
        ## forward + backprop + loss

        # clear old gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        # compute loss
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()

        ## update model params
        optimizer.step()

    model.eval()
    acc_hist['train'].append(get_accuracy(model, train_loader))
    acc_hist['val'].append(get_accuracy(model, val_loader))
    acc_hist['test'].append(get_accuracy(model, test_loader))

    if verbose:
      print('Epoch: %d | Train Accuracy: %.2f | Validation Accuracy: %.2f | Test Accuracy: %.2f' \
           %(epoch, acc_hist['train'][-1], acc_hist['val'][-1], acc_hist['test'][-1]))

  return model, acc_hist

def experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.5, weight_decay=0.01, num_epochs=50, verbose=False):
  # Use GPU if it is available.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Inpute Batch size:
  BATCH_SIZE = 32

  # Convert images to tensor
  transform = transforms.Compose(
      [transforms.ToTensor()])

  # Get train, validation and test data loader.
  trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

  train_data = notMNIST(trainTarget, trainData, transform=transform)
  val_data = notMNIST(validTarget, validData, transform=transform)
  test_data = notMNIST(testTarget, testData, transform=transform)


  train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

  # Specify which model to use
  if model_type == 'CNN':
    model = CNN(dropout_rate)
  elif model_type == 'FNN':
    model = FNN(dropout_rate)


  # Loading model into device
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  model, acc_hist = train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=num_epochs, verbose=verbose)

  # Release the model from the GPU (else the memory wont hold up)
  model.cpu()

  return model, acc_hist

def compare_arch():
    # train CNN with the required hyperparameters
    cnn_model, cnn_hist = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.0,
        weight_decay=0.0,
        num_epochs=50,
        verbose=True
    )

    # train FNN with the same hyperparameters
    fnn_model, fnn_hist = experiment(
        model_type='FNN',
        learning_rate=0.0001,
        dropout_rate=0.0,
        weight_decay=0.0,
        num_epochs=50,
        verbose=True
    )

    # create an epoch index for plotting
    epochs = range(1, 51)

    # plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cnn_hist['train'], label='CNN Train')
    plt.plot(epochs, fnn_hist['train'], label='FNN Train')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy: CNN vs FNN')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot testing accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cnn_hist['test'], label='CNN Test')
    plt.plot(epochs, fnn_hist['test'], label='FNN Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy: CNN vs FNN')
    plt.legend()
    plt.grid(True)
    plt.show()

    # print final accuracies for easier comparison
    print("Final CNN Train Accuracy: {:.2f}%".format(cnn_hist['train'][-1]))
    print("Final CNN Test Accuracy:  {:.2f}%".format(cnn_hist['test'][-1]))

    print("Final FNN Train Accuracy: {:.2f}%".format(fnn_hist['train'][-1]))
    print("Final FNN Test Accuracy:  {:.2f}%".format(fnn_hist['test'][-1]))

def compare_dropout():
    # train CNN with dropout rate of 0.5
    model_05, hist_05 = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.5,
        weight_decay=0.0,
        num_epochs=50,
        verbose=True
    )

    # train CNN with dropout rate of 0.8
    model_08, hist_08 = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.8,
        weight_decay=0.0,
        num_epochs=50,
        verbose=True
    )

    # train CNN with dropout rate of 0.95
    model_095, hist_095 = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.95,
        weight_decay=0.0,
        num_epochs=50,
        verbose=True
    )

    # create an epoch index for plotting
    epochs = range(1, 51)

    # plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, hist_05['train'], label='Dropout = 0.5')
    plt.plot(epochs, hist_08['train'], label='Dropout = 0.8')
    plt.plot(epochs, hist_095['train'], label='Dropout = 0.95')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Training Accuracy for Different Dropout Rates')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot testing accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, hist_05['test'], label='Dropout = 0.5')
    plt.plot(epochs, hist_08['test'], label='Dropout = 0.8')
    plt.plot(epochs, hist_095['test'], label='Dropout = 0.95')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Testing Accuracy for Different Dropout Rates')
    plt.legend()
    plt.grid(True)
    plt.show()

    # print final accuracies for easier comparison
    print("Final Train Accuracy with dropout 0.5: {:.2f}%".format(hist_05['train'][-1]))
    print("Final Test Accuracy with dropout 0.5:  {:.2f}%".format(hist_05['test'][-1]))

    print("Final Train Accuracy with dropout 0.8: {:.2f}%".format(hist_08['train'][-1]))
    print("Final Test Accuracy with dropout 0.8:  {:.2f}%".format(hist_08['test'][-1]))

    print("Final Train Accuracy with dropout 0.95: {:.2f}%".format(hist_095['train'][-1]))
    print("Final Test Accuracy with dropout 0.95: {:.2f}%".format(hist_095['test'][-1]))

def compare_l2():
    # train CNN with weight decay = 0.1
    model_01, hist_01 = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.0,
        weight_decay=0.1,
        num_epochs=50,
        verbose=True
    )

    # train CNN with weight decay = 1.0
    model_10, hist_10 = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.0,
        weight_decay=1.0,
        num_epochs=50,
        verbose=True
    )

    # train CNN with weight decay = 10.0
    model_100, hist_100 = experiment(
        model_type='CNN',
        learning_rate=0.0001,
        dropout_rate=0.0,
        weight_decay=10.0,
        num_epochs=50,
        verbose=True
    )

    # create an epoch index for plotting
    epochs = range(1, 51)

    # plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, hist_01['train'], label='weight_decay = 0.1')
    plt.plot(epochs, hist_10['train'], label='weight_decay = 1.0')
    plt.plot(epochs, hist_100['train'], label='weight_decay = 10.0')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Training Accuracy for Different Weight Decay Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot testing accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, hist_01['test'], label='weight_decay = 0.1')
    plt.plot(epochs, hist_10['test'], label='weight_decay = 1.0')
    plt.plot(epochs, hist_100['test'], label='weight_decay = 10.0')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Testing Accuracy for Different Weight Decay Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    # print final results
    print("Final Train Accuracy with weight decay 0.1: {:.2f}%".format(hist_01['train'][-1]))
    print("Final Test Accuracy with weight decay 0.1:  {:.2f}%".format(hist_01['test'][-1]))

    print("Final Train Accuracy with weight decay 1.0: {:.2f}%".format(hist_10['train'][-1]))
    print("Final Test Accuracy with weight decay 1.0:  {:.2f}%".format(hist_10['test'][-1]))

    print("Final Train Accuracy with weight decay 10.0: {:.2f}%".format(hist_100['train'][-1]))
    print("Final Test Accuracy with weight decay 10.0: {:.2f}%".format(hist_100['test'][-1]))

# run experiments
'''
compare_arch()
compare_dropout()
compare_l2()
'''