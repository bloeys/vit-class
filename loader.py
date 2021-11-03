import h5py
import torch
import numpy as np

dataDir = "./vit-class/data/"


class PCamDataset(torch.utils.data.Dataset):

    def __init__(self, dataFile, labelsFile, dataKey, labelKey):

        super(PCamDataset, self).__init__()

        hfFile = h5py.File(dataFile, 'r')
        self.data = np.array(hfFile.get(dataKey))
        hfFile.close()

        hfFile = h5py.File(labelsFile, 'r')
        self.labels = np.array(hfFile.get(labelKey))
        hfFile.close()

    def __getitem__(self, index):

        return (
            #Channels are last (96,96,3) but we need it to be first (3,96,96) so move it around
            torch.from_numpy(self.data[index]).float().permute(2, 0, 1),
            #Labels is in the form [[[0]]], but we only need the value, so we use squeeze
            torch.from_numpy(np.squeeze(self.labels[index])))

    def __len__(self):
        return len(self.data)


def GetTrainLoader():

    print('\nloading training data...')
    trainData = PCamDataset(dataDir+'camelyonpatch_level_2_split_train_x.h5',
                            dataDir+'camelyonpatch_level_2_split_train_y.h5', 'x', 'y')
    print('training data loaded with', trainData.__len__(), 'elements')

    # num_workers must be zero otherwise we get an error
    trainLoader = torch.utils.data.DataLoader(
        trainData, batch_size=64, shuffle=True, num_workers=0)

    return trainLoader


def GetTestLoader():

    print('\nloading test data...')
    testData = PCamDataset(dataDir+'camelyonpatch_level_2_split_test_x.h5',
                           dataDir+'camelyonpatch_level_2_split_test_y.h5', 'x', 'y')
    print('test data loaded with', testData.__len__(), 'elements')

    testLoader = torch.utils.data.DataLoader(
        testData, batch_size=64, shuffle=True, num_workers=0)

    return testLoader


def GetValidationLoader():

    print('\nloading validation data...')
    validData = PCamDataset(dataDir+'camelyonpatch_level_2_split_test_x.h5',
                            dataDir+'camelyonpatch_level_2_split_test_y.h5', 'x', 'y')
    print('validation data loaded with', validData.__len__(), 'elements')

    validLoader = torch.utils.data.DataLoader(
        validData, batch_size=64, shuffle=True, num_workers=0)

    return validLoader
