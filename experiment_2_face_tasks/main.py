
RANDOM_SEED = 13011


# Python imports.
import os
import itertools
from collections import namedtuple

# Randomness imports.
import random
random.seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)

# Torch imports.
import torch
torch.manual_seed(RANDOM_SEED)
import torch.nn as nn
from torchvision import datasets as ds
from torchvision import transforms as ts
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils import data
from torch.utils.data import DataLoader

# Custom imports.
from Doric import ProgNet
from model_generator import VariationalAutoEncoderModelGenerator as FaceVAEGen
from tasknet import TaskNet

# Constants.
TASKS = ["reconstruction", "denoise", "colorize", "inpaint"]
USE_CPU = False
BATCH_SIZE = 24
TEST_N = BATCH_SIZE * 2
EPOCHS = [0, 0, 0, 50]
LR = 0.001
KL_WEIGHT = 0.00003
Z_DIM = 128

OUTPUT_DIR = "./data/outputs/"
SAVE_DIR = "./data/models/"
MASK_DIR = "./data/mask/"
DATA_DIR_TRAIN = "./data/celeba_small_train/"
DATA_DIR_VALID = "./data/celeba_small_valid/"
DATA_DIR_TEST = "./data/celeba_small_test/"

MASKS = DataLoader(ds.ImageFolder(MASK_DIR, ts.ToTensor()), batch_size = BATCH_SIZE, drop_last = True)
MASK_CYCLE = itertools.cycle(MASKS)

TRAIN_PRINT_AT = 14
VALID_PRINT_AT = 1
TEST_PRINT_AT = 1

DEVICE = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda:0")

temp = None
temp2 = None
temp3 = None



def transformInput(x, method):
    global temp
    global temp2
    global temp3
    if method == 'denoise':
        return x + torch.randn(*x.shape) * 0.1
    elif method == 'colorize':
        img = x.mean(axis = 1, keepdim = True)
        img = torch.cat((img, img, img), dim = 1)
        return img
    elif method == 'inpaint':
        mask, v = next(MASK_CYCLE)
        temp = v
        temp2 = mask
        temp3 = x
        return x * mask
    else:
        return x

def saveTasknet(tasknet):
    filename = "model_params.pt"
    filepath = os.path.join(SAVE_DIR, filename)
    torch.save(tasknet.prognet.state_dict(), filepath)
    #tasknet.detector.saveAll()

def trainHelper(model, modelGen, col, optimizer, x, task):
    with torch.no_grad():
        xOriginal = x.to(DEVICE)
    x = transformInput(x, task).to(DEVICE)
    xReconst = model(col, x)
    reconstLoss = F.mse_loss(xReconst, xOriginal)
    mu, var = modelGen.getLastVarBlock().getDistribution()
    print(mu, var)
    klDivergence = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
    loss = reconstLoss + KL_WEIGHT * klDivergence
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)

def getDataAndPreprocess():
    transform = ts.Compose([ts.RandomHorizontalFlip(), ts.CenterCrop(148), ts.Resize(64), ts.ToTensor()])
    datasetTrain = ds.ImageFolder(DATA_DIR_TRAIN, transform)
    dataLoaderTrain = DataLoader(datasetTrain, batch_size = BATCH_SIZE, drop_last = True)
    datasetValid = ds.ImageFolder(DATA_DIR_VALID, transform)
    dataLoaderValid = DataLoader(datasetValid, batch_size = BATCH_SIZE, drop_last = True)
    datasetTest = ds.ImageFolder(DATA_DIR_TEST, transform)
    dataLoaderTest = DataLoader(datasetTest, batch_size = BATCH_SIZE, drop_last = True)
    return (dataLoaderTrain, dataLoaderValid, dataLoaderTest)

def buildTaskNet():
    prognetGen = FaceVAEGen(Z_DIM)
    prognet = ProgNet(colGen = prognetGen)
    tasknet = TaskNet(prognet, None, prognetGen)   #TODO: Add detector.
    return tasknet

def train(tasknet, trainDS, validDS):
    print("Training and validation started.")
    model = tasknet.prognet
    modelGen = tasknet.prognetGen
    detector = tasknet.detector
    for k, task in enumerate(TASKS):
        print("Task %d (%s) started." % (k, task))
        model.freezeAllColumns()
        col = model.addColumn(msg = task)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)
        for epoch in range(EPOCHS[k]):
            print("  Running epoch %d." % epoch)
            print("    [", end = '', flush = True)
            avgLoss = 0.0
            for i, (x, _) in enumerate(trainDS):
                loss = trainHelper(model, modelGen, col, optimizer, x, task)
                avgLoss += loss
                if i % TRAIN_PRINT_AT == 0:
                    print("=", end = '', flush = True)
            print("]")
            print("    [", end = '', flush = True)
            for i, (x, _) in enumerate(validDS):
                with torch.no_grad():
                    xOriginal = x.to(DEVICE)
                    x = transformInput(x, task).to(DEVICE)
                    xReconst = model(col, x)
                    xConcat = torch.cat([xOriginal.cpu().data, x.cpu().data, xReconst.cpu().data], dim = 3)
                    filename = "task-%s-epoch-%d-i-%d.png" % (task, epoch, i)
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    save_image(xConcat, filepath)
                    if i % VALID_PRINT_AT == 0:
                        print("#", end = '', flush = True)
            #torch.cuda.empty_cache()
            print("]")
            if task == "inpaint":
                print("Safety save.")
                saveTasknet(tasknet)
            print("    Average training loss: %f." % avgLoss)
    print("Training and validation done.")

def test(tasknet, testDS):
    print("Testing started.")
    model = tasknet.prognet
    modelGen = tasknet.prognetGen
    detector = tasknet.detector
    for k, task in enumerate(TASKS):
        print("Task %d (%s) started." % (k, task))
        col = task
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)
        print("    [", end = '', flush = True)
        for i, (x, _) in enumerate(testDS):
            with torch.no_grad():
                xOriginal = x.to(DEVICE)
                x = transformInput(x, task).to(DEVICE)
                xReconst = model(col, x)
                xConcat = torch.cat([xOriginal.cpu().data, x.cpu().data, xReconst.cpu().data], dim = 3)
                filename = "task-%s-i-%d.png" % (task, i)
                filepath = os.path.join(OUTPUT_DIR, filename)
                save_image(xConcat, filepath)
                if i % TEST_PRINT_AT == 0:
                    print("*", end = '', flush = True)
            torch.cuda.empty_cache()
            print("]")
    print("Testing done.")




def main():
    print("Using device:  %s." % str(DEVICE))
    print("Data sources:  %s & %s & %s." % (DATA_DIR_TRAIN, DATA_DIR_VALID, DATA_DIR_TEST))
    trainData, validData, testData = getDataAndPreprocess()
    print("Training dataset size:    %d." % len(trainData.dataset))
    print("Validation dataset size:  %d." % len(validData.dataset))
    print("Testing dataset size:     %d." % len(testData.dataset))
    tasknet = buildTaskNet()
    saveTasknet(tasknet)
    train(tasknet, trainData, validData)
    saveTasknet(tasknet)
    test(tasknet, testData)
    print("Done.")




if __name__ == '__main__':
    main()

#===============================================================================
