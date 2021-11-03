import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import loader
import model as mdl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

classes = ['no_cancer', 'cancer']

trainLoader = loader.GetTrainLoader()
testLoader = loader.GetTestLoader()
validLoader = loader.GetValidationLoader()

model = mdl.GetNewModel()
# model = mdl.LoadModel('./vit-class/trained-model')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)


def Train(epoch):

    for i, data in enumerate(trainLoader):

        imgBatch, labelBatch = data
        imgBatch = imgBatch.to(device)
        labelBatch = labelBatch.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(imgBatch)

        loss = criterion(outputs, labelBatch)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch '+str(epoch)+' finished '+str(i)+' batches')


def Validate():

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    total = 0
    correct = 0
    with torch.no_grad():
        for data in validLoader:
            imgBatch, labelBatch = data
            imgBatch = imgBatch.to(device)
            labelBatch = labelBatch.to(device)

            # calculate outputs by running images through the network
            outputs = model(imgBatch)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labelBatch.size(0)
            correct += (predicted == labelBatch).sum().item()

            # collect the correct predictions for each class
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labelBatch, predictions):

                total_pred[classes[label]] += 1
                if label == prediction:
                    correct_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))
    # Print total accuracy
    print('Accuracy of the network on the 32,768 validation images: %d %%' %
          (100 * correct / total))


epochsToRunFor = 10
for epoch in range(epochsToRunFor):
    Train(epoch+1)
    Validate()
    print('===epoch', epoch+1, 'done===\n')

mdl.SaveModel(model, './vit-class/trained-model-'+str(epochsToRunFor))
