import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import loader
import model as myModel
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

shouldLoadModel = False
classes = ['no_cancer', 'cancer']

trainLoader = loader.GetTrainLoader()
testLoader = loader.GetTestLoader()
validLoader = loader.GetValidationLoader()

model = myModel.PCamNet().to(device)

def SaveModel(model, modelPath):
    torch.save(model.state_dict(), modelPath)

def LoadModel(modelPath):
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    print("Loaded model", modelPath)

def ShowImgs(img):
    
    if img.__len__ == None:
        plt.imshow(img.detach().clone().to("cpu").permute(1, 2, 0))
        plt.show()
        return

    numImgs = len(img)
    rows = cols = numImgs//2

    fig = plt.figure()
    for i in range(len(img)):

        fig.add_subplot(rows, cols, i+1)
        
        plt.imshow(img[i].detach().clone().to("cpu").permute(1, 2, 0))
        plt.axis('off')
        plt.title("Img "+str(i+1))

    plt.show()

if shouldLoadModel:
    LoadModel('./dense201-stn/trained/no-stn/dn201-stn-pre-0.011218384839594364')

z = 0
bestLoss = 99999
lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

lossFile = open('./dense201-stn/trained/original-dn/loss-pre-orig-Affine-Stn-'+str(time())+'.txt', 'w')

def Train(epoch):

    global z
    global bestLoss

    #Enable weight updates to allow training
    model.train()

    avg100Loss = 0.0
    for i, data in enumerate(trainLoader):

        z += 1

        imgBatch, labelBatch = data
        imgBatch = imgBatch.to(device)

        labelBatch = labelBatch.to(device).float()
        labelBatch = labelBatch.type('torch.LongTensor').to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(imgBatch)

        loss = lossFunc(outputs, labelBatch)
        loss.backward()

        optimizer.step()

        avg100Loss += float(loss)

        #If best loss so far store the model
        if loss < bestLoss:
            bestLoss = float(loss)
            if loss < 1:
                SaveModel(model, './dense201-stn/trained/original-dn/dn201-stn-pre-orig-Affine-Stn-'+str(float(loss)))

        lossFile.write(str(float(loss))+'\n')

        if i > 0 and i % 100 == 0:
            print('Epoch '+str(epoch)+' finished '+str(i)+' batches. Avg. loss of last 100 batches: '+str(float(avg100Loss / 100)))
            avg100Loss = 0.0
            lossFile.flush()


def Validate():

    global z
    model.eval()

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    total = 0
    correct = 0

    #Positive = Cancer; Negative = NoCancer
    #TP = Cancer->Cancer; TN = NoCancer->NoCancer
    #FP = NoCancer->Cancer; FN = Cancer->NoCancer
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for data in validLoader:

            z += 1
            imgBatch, labelBatch = data
            imgBatch = imgBatch.to(device)
            labelBatch = labelBatch.to(device)

            # calculate outputs by running images through the network
            outputs = model(imgBatch)

            # the class with the highest energy is what we choose as prediction
            labelBatch = labelBatch.type('torch.LongTensor').to(device)

            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labelBatch)):

                #Real is NoCancer
                if labelBatch[i] == 0:
                    if predicted[i] == 0:
                        tn += 1
                    else:
                        fp += 1
                #Real is Cancer
                else:
                    if predicted[i] == 0:
                        fn +=1
                    else:
                        tp += 1

            total += labelBatch.size(0)
            correct += (predicted == labelBatch).sum().item()

            # collect the correct predictions for each class
            for label, prediction in zip(labelBatch, predicted):

                total_pred[classes[label]] += 1
                if label == prediction:
                    correct_pred[classes[label]] += 1

            if z%100==0:
                print('Finished',str(z*64)+'/32,768 validation images')

    z=0
    print('tp:',tp)
    print('tn:',tn)
    print('fp:',fp)
    print('fn:',fn)

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))
    # Print total accuracy
    print('Accuracy of the network on the 32,768 validation images: %d %%' %
          (100 * correct / total))


writer = SummaryWriter('./logdir')

epochsToRunFor = 40
for epoch in range(epochsToRunFor):
    Train(epoch+1)
    Validate()
    print('===epoch', epoch+1, 'done===\n')

lossFile.close()

###2-out###
#STN
# tp: 10839
# tn: 16090
# fp: 301
# fn: 5538

#No STN
# tp: 10703
# tn: 16068
# fp: 323
# fn: 5674