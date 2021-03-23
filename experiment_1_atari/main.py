

import random
import math
from collections import defaultdict

from .multienv import MultiEnv
from .random_agent import RandomAgent
from .task_detector import TaskDetector
#from .task_detector import AnomalyDetectorGenerator
from .autoencoder import AutoEncoder
from .vae import VAE
from .vae import Encoder as VAEEncoder
from .vae import Decoder as VAEDecoder

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image as Image

GPU_TRAINING_ON = True
TRAIN_RECS = 1000
TRAIN_EPOCHS = 1
TEST_RECS = 100

NN_SIZE = (77, 100)
H_DIM = 300
Z_DIM = 128

#MULTI_GPU_TESTING = False
#MULTI_GPU_LIST = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]
#DETECTORS_PER_CARD = 2

DEF_ENVS = ["breakout", "pong", "space_invaders", "ms_pacman", "assault", "asteroids", "boxing", "phoenix", "alien"]

device = None

SAMPLES_DIR = 'samples'
MODELS_DIR = 'models'




class DetGen:
    def __init__(self, aeMode):
        super().__init__()
        self.aeMode = aeMode

    def generateDetector(self):
        if self.aeMode == "vae":
            vae = buildVAE()
            return vae.to(device)
        else:
            return AutoEncoder(NN_SIZE, H_DIM, Z_DIM).to(device)



def buildVAE():
    enc = VAEEncoder(NN_SIZE, Z_DIM, Z_DIM, h = H_DIM)
    dec = VAEDecoder(Z_DIM, NN_SIZE, h = H_DIM)
    model = VAE(enc, dec)
    return model



def preprocess(inputDict):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(NN_SIZE, Image.NEAREST),
        lambda x: transforms.functional.vflip(x),
        transforms.ToTensor(),
    ])
    x = inputDict['S0'].T
    x = transform(x)
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    n_x = inputDict['S1'].T
    n_x = transform(n_x)
    n_x = torch.unsqueeze(n_x, dim=0)
    n_x = n_x.to(device).detach()
    return x, n_x



def convertTorch(state):
    return torch.from_numpy(state)



'''
def enableMultiGPUTesting(detector):
    cacheSize = DETECTORS_PER_CARD * len(MULTI_GPU_LIST)
    detector.resetCache(cacheSize)
    perCardCount = 0
    card = 0
    for name in list(detector.cache):
        dev = torch.device(MULTI_GPU_LIST[card])
        detector.resetDetectordevice(dev)
        perCardCount += 1
        if perCardCount >= DETECTORS_PER_CARD:
            card += 1
            perCardCount = 0
'''



def test(agent, detector, env, envID, ds):
    predicteds = defaultdict(lambda: 0.0)
    predictedsNorm = defaultdict(lambda: 0.0)
    with torch.no_grad():
        for inpNum, inp in enumerate(ds):
            x, n_x = preprocess(inp)
            envPred, envPredNorm = detector.detect(x)
            predicteds[(envPred, envID)] += 1
            predictedsNorm[(envPredNorm, envID)] += 1
            print("Env: {}   Record: {}/{}   Corr: {}={}-->{}   NCorr: {}={}-->{}".format(envID, inpNum, len(ds), envPred, envID, (envPred == envID), envPredNorm, envID, (envPredNorm == envID)))
    return predicteds, predictedsNorm





def genDataFromEnv(agent, env, datasetSize, render = False):
    ds = []
    while True:
        state = convertTorch(env.reset())
        terminal = False
        i = 0
        while not terminal:
            if render:   env.render()
            action = agent.act(state)
            nextState, reward, terminal, info = env.step(action)
            nextState = convertTorch(nextState)
            detectorInput = {"S0": state, "S1": nextState, "A": action}
            ds.append(detectorInput)
            state = nextState
            i = i + 1
        if len(ds) >= datasetSize:
            ds = random.sample(ds, datasetSize)
            return ds





def train(agent, detector, env, envID, ds, epochs, sampleDir):
    for epoch in range(epochs):
        for inpNum, inp in enumerate(ds):
            out, loss = trainDetector(detector, inp, envID)
            print("Env: {}   Epoch: {}/{}   Record: {}/{}   Loss: {}".format(envID, epoch, epochs, inpNum, len(ds), loss))
        if sampleDir[-1] == '/':
            img_path = "{}env_{}_e_{}.png".format(sampleDir, envID, epoch)
        else:
            img_path = "{}/env_{}_e_{}.png".format(sampleDir, envID, epoch)
        save_image(torch.rot90(out, 3, [2, 3]), img_path)
    for inpNum, inp in enumerate(ds):
        trainDetectorDistro(detector, inp, envID)



'''
def train(agent, detector, env, envNumber, episodes, distroEpisodes, render = False):
    for e in range(episodes):
        state = env.reset()
        state = convertTorch(state)
        terminal = False
        i = 0
        while not terminal:
            if render:   env.render()
            action = agent.act(state)
            nextState, reward, terminal, info = env.step(action)
            nextState = convertTorch(nextState)
            detectorInput = {"S0": state, "S1": nextState, "A": action}
            out, loss = trainDetector(detector, detectorInput, envNumber)
            print("Env: {} Episode: {}/{} StateNo: {} Loss: {}".format(envNumber, e, episodes, i, loss))
            state = nextState
            i = i + 1
        img_path = "%s/env_" % SAMPLES_DIR + str(envNumber) + "_e_" + str(e) +"_img" + ".png"
        save_image(torch.rot90(out, 3, [2,3]),img_path)
    for e in range(distroEpisodes):
        state = env.reset()
        state = convertTorch(state)
        terminal = False
        while not terminal:
            if render:   env.render()
            action = agent.act(state)
            nextState, reward, terminal, info = env.step(action)
            nextState = convertTorch(nextState)
            detectorInput = {"S0": state, "S1": nextState, "A": action}
            trainDetectorDistro(detector, detectorInput, envNumber)
            state = nextState
'''


def trainDetectorDistro(detector, inputDict, envLabel):
    with torch.no_grad():
        x, n_x = preprocess(inputDict)
        detector.trainDistro(x, envLabel)



def trainDetector(detector, inputDict, envLabel):
    x, n_x = preprocess(inputDict)
    return detector.trainStep(x, n_x, envLabel)


def populateCM(taskList, cm, predicteds):
    for trueEnv in taskList:
        for predEnv in taskList:
            cm[(predEnv, trueEnv)] += predicteds[(predEnv, trueEnv)]



def printCM(taskList, cm):
    print("          Predicted:")
    print("         ", end = '')
    dotsLen = 0
    for task in taskList:
        s = task + "   "
        print(s, end = '')
        dotsLen += len(s)
    print()
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()
    for i, trueEnv in enumerate(taskList):
        print("Task " + str(i) + ":  |", end = '')
        for predEnv in taskList:
            print("{:8.3f}".format(cm[(predEnv, trueEnv)]), end = ' ')
        print("|")
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()



def configCLIParser(parser):
    parser.add_argument("--train_size", help="Number of records to generate for training.", type=int, default=TRAIN_RECS)
    parser.add_argument("--train_epochs", help="Training epochs.", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--test_size", help="Number of records to generate for testing.", type=int, default=TEST_RECS)
    parser.add_argument("--train_mode", help="If 2, train on one enviroment specified by train_env. If 1, trains all detectors. If 0, attempts to load detectors.", choices=[0, 1, 2], type=int, default=1)
    parser.add_argument("--train_env", help="Env to use if train_mode is set to 2.", choices=DEF_ENVS, default=DEF_ENVS[0])
    parser.add_argument("--test_mode", help="If 1, tests the detectors. If 0, skips testing.", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ae_type", help="Type of AE to use.", choices=["vae", "ae"], default="vae")
    parser.add_argument("--device", help="Device to run torch on. Usually 'cpu' or 'cuda:[N]'. Defaults to cpu if cuda is not available.", type=str, default="cpu")
    parser.add_argument("--models_dir", help="Directory to store model save / load files.", type=str, default = "./%s/" % MODELS_DIR)
    parser.add_argument("--samples_dir", help="Directory to store training reconst samples.", type=str, default = "./%s/" % SAMPLES_DIR)
    return parser



def main(args):
    global device
    print("Starting.")
    if torch.cuda.is_available():
        print("Cuda is available.")
        print("Using device: %s." % args.device)
        device = torch.device(args.device)
    else:
        print("Cuda is not available.")
        print("Using device: cpu.")
        device = torch.device("cpu")
    if args.train_mode == 2:
        envNameList = [args.train_env]
    else:
        envNameList = DEF_ENVS
    atariGames = MultiEnv(envNameList)
    agent = RandomAgent(atariGames.actSpace)
    gen = DetGen(args.ae_type)
    taskDetector = TaskDetector(gen, args.models_dir)
    if args.train_mode > 0:
        for i, env in enumerate(atariGames.getEnvList()):
            ds = genDataFromEnv(agent, env, args.train_size)
            taskDetector.addTask(env.game)
            train(agent, taskDetector, env, env.game, ds, args.train_epochs, args.samples_dir)
            taskDetector.expelDetector(env.game)
        print("Training complete.\n\n")
    else:
        taskDetector.loadAll(envNameList)
        print("Loaded envs %s." % str(envNameList))
    if args.test_mode == 1:
        print("Testing with and without normalization.")
        cm = defaultdict(lambda: 0.0)
        cmn = defaultdict(lambda: 0.0)
        for i, env in enumerate(atariGames.getEnvList()):
            ds = genDataFromEnv(agent, env, args.test_size)
            predicteds, predictedsNorm = test(agent, taskDetector, env, env.game, ds)
            populateCM(envNameList, cm, predicteds)
            populateCM(envNameList, cmn, predictedsNorm)
        print("Testing complete.\n")
        print("Not normalized:\n\n")
        printCM(envNameList, cm)
        print("\n\nNormalized:\n\n")
        printCM(envNameList, cmn)
    print("\n\nDone.")




if __name__ == '__main__':
    main()

#===============================================================================
