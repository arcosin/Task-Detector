
RANDOM_SEED = 13011


# Python imports.
import os
import itertools
from collections import namedtuple, defaultdict

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
from .Doric import ProgNet
from .task_detector import TaskDetector
from .autoencoder import AutoEncoder
from .vae import VAE
from .vae import Encoder as VAEEncoder
from .vae import Decoder as VAEDecoder

# Constants.
TASKS = ["reconstruction", "denoise", "colorize", "inpaint"]
USE_CPU = False
BATCH_SIZE = 24
TEST_BATCH_SIZE = 8
TEST_N = BATCH_SIZE * 2
EPOCHS = {"reconstruction": 20, "denoise": 20, "colorize": 20, "inpaint": 20}
Z_DIM = 64
NN_SIZE = (64, 64)
H_DIM = 300

TRAIN_ENABLED = True
OUTPUT_DIR = "./data/outputs/"
SAVE_DIR = "./data/models/"
MASK_DIR = "./data/mask/"
DATA_DIR_TRAIN = "./data/celeba_small_train/"
DATA_DIR_VALID = "./data/celeba_small_valid/"
DATA_DIR_TEST = "./data/celeba_small_test/"

MASKS = DataLoader(ds.ImageFolder(MASK_DIR, ts.ToTensor()), batch_size = BATCH_SIZE, drop_last = True)
MASK_CYCLE = itertools.cycle(MASKS)
TEST_MASKS = DataLoader(ds.ImageFolder(MASK_DIR, ts.ToTensor()), batch_size = TEST_BATCH_SIZE, drop_last = True)
TEST_MASK_CYCLE = itertools.cycle(TEST_MASKS)

TRAIN_PRINT_AT = 14
VALID_PRINT_AT = 1
TEST_PRINT_AT = 1

AE_MODE = "vae"                    # Options: "ae", "vae".

DEVICE = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda:0")



class DetGen:
    def __init__(self):
        super().__init__()

    def generateDetector(self):
        if AE_MODE == "vae":
            vae = buildVAE()
            return vae.to(DEVICE)
        else:
            return AutoEncoder(NN_SIZE, H_DIM, Z_DIM).to(DEVICE)



def buildVAE():
    enc = VAEEncoder(NN_SIZE, Z_DIM, Z_DIM, h = H_DIM)
    dec = VAEDecoder(Z_DIM, NN_SIZE, h = H_DIM)
    model = VAE(enc, dec)
    return model



def transformInput(x, method, masks = MASK_CYCLE):
    if method == 'denoise':
        return x + torch.randn(*x.shape) * 0.1
    elif method == 'colorize':
        img = x.mean(axis = 1, keepdim = True)
        img = torch.cat((img, img, img), dim = 1)
        return img
    elif method == 'inpaint':
        mask, v = next(masks)
        return x * mask
    else:
        return x


def printCM(cm):
    print("          Predicted:")
    print("         ", end = '')
    dotsLen = 0
    for task in TASKS:
        s = task + "   "
        print(s, end = '')
        dotsLen += len(s)
    print()
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()
    for taskNum in range(len(TASKS)):
        print("Task " + str(taskNum) + ":  |", end = '')
        for predNum in range(len(TASKS)):
            print("{:8.3f}".format(cm[(TASKS[predNum], TASKS[taskNum])]), end = "   ")
        print("|")
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()


def getDataAndPreprocess():
    transform = ts.Compose([ts.RandomHorizontalFlip(), ts.CenterCrop(148), ts.Resize(64), ts.ToTensor()])
    datasetTrain = ds.ImageFolder(DATA_DIR_TRAIN, transform)
    dataLoaderTrain = DataLoader(datasetTrain, batch_size = BATCH_SIZE, drop_last = True)
    datasetValid = ds.ImageFolder(DATA_DIR_VALID, transform)
    dataLoaderValid = DataLoader(datasetValid, batch_size = BATCH_SIZE, drop_last = True)
    datasetTest = ds.ImageFolder(DATA_DIR_TEST, transform)
    dataLoaderTest = DataLoader(datasetTest, batch_size = TEST_BATCH_SIZE, drop_last = True)
    return (dataLoaderTrain, dataLoaderValid, dataLoaderTest)



def buildTaskDetector(saveDir):
    gen = DetGen()
    taskDetector = TaskDetector(gen, saveDir)
    return taskDetector



def train(detector, task, trainDS):
    print("Training started on task %s." % task)
    for epoch in range(EPOCHS[task]):
        print("  Running epoch %d." % epoch)
        print("    [", end = '', flush = True)
        avgLoss = 0.0
        for i, (x, _) in enumerate(trainDS):
            x = transformInput(x, task).to(DEVICE)
            _, loss = detector.trainStep(x, task)
            if isinstance(loss, dict):
                avgLoss += loss["loss"]
            else:
                avgLoss += loss
            if i % TRAIN_PRINT_AT == 0:
                print("=", end = '', flush = True)
        print("]")
    detector.expelDetector(task)
    print("Training finished on task %s." % task)



def test(detector, task, testDS, cm = None, cmn = None):
    if cm is None:
        cm = defaultdict(lambda: 0.0)
    if cmn is None:
        cmn = defaultdict(lambda: 0.0)
    print("Testing started on task %s." % task)
    for i, (x, _) in enumerate(testDS):
        x = transformInput(x, task, TEST_MASK_CYCLE).to(DEVICE)
        predTask, predTaskNormalized = detector.detect(x)
        cm[(predTask, task)] += 1
        cmn[(predTaskNormalized, task)] += 1
        print("Env: {}  Image: {}/{}  Pred: {}  Correct: {}".format(task, i, len(testDS), predTask, (predTask == task)))
    print("Testing finished on task %s." % task)
    return (cm, cmn)



def configCLIParser(parser):
    #parser.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--save_dir", help="Specify where to save models.", type=str, default=SAVE_DIR)
    return parser



def main(args):
    dataTrain, dataValid, dataTest = getDataAndPreprocess()
    detector = buildTaskDetector(args.save_dir)
    cm = defaultdict(lambda: 0.0)
    cmn = defaultdict(lambda: 0.0)
    for task in TASKS:
        detector.addTask(task)
        if TRAIN_ENABLED:
            train(detector, task, dataTrain)
            detector.saveDetector(task)
        else:
            detector.rebuildDetector(task)
    for task in TASKS:
        cm, cmn = test(detector, task, dataTest, cm = cm, cmn = cmn)
    print("Testing batch size: %s." % TEST_BATCH_SIZE)
    print("\n\n")
    print("Confusion matrix.")
    printCM(cm)
    print("\n\n")
    print("Confusion matrix with normalize.")
    printCM(cmn)
    print("Done.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "task_detector_face_experiment", description = "")
    parser = configCLIParser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
