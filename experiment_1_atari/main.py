


from collections import defaultdict
from multienv import MultiEnv
from random_agent import RandomAgent
from task_detector import TaskDetector

from task_detector import AnomalyDetectorGenerator

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image as Image

TRAINING_ON = True
GPU_TRAINING_ON = True
TRAIN_EPS = 100
DISTRO_TRAIN_EPS = 5
TEST_EPS = 10
NN_SIZE = (77, 100)
H_DIM = 300
LATENT_SIZE = 128

MULTI_GPU_TESTING = False
MULTI_GPU_LIST = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]
DETECTORS_PER_CARD = 2

DEVICE = torch.device("cuda:0" if GPU_TRAINING_ON and torch.cuda.is_available() else "cpu")

SAMPLES_DIR = 'samples'
MODELS_DIR = 'models'

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
    x = x.to(DEVICE)

    n_x = inputDict['S1'].T
    n_x = transform(n_x)
    n_x = torch.unsqueeze(n_x, dim=0)
    n_x = n_x.to(DEVICE).detach()

    return x, n_x

def convertTorch(state):
    return torch.from_numpy(state)

def enableMultiGPUTesting(detector):
    cacheSize = DETECTORS_PER_CARD * len(MULTI_GPU_LIST)
    detector.resetCache(cacheSize)
    perCardCount = 0
    card = 0
    for name in list(detector.cache):
        dev = torch.device(MULTI_GPU_LIST[card])
        detector.resetDetectorDevice(dev)
        perCardCount += 1
        if perCardCount >= DETECTORS_PER_CARD:
            card += 1
            perCardCount = 0

def test(agent, detector, env, envNumber, episodes, render = False):
    predicteds = defaultdict(lambda: 0.0)
    if MULTI_GPU_TESTING:
         enableMultiGPUTesting(detector)
    with torch.no_grad():
        for e in range(episodes):
            state = env.reset()
            state = convertTorch(state)
            terminal = False
            i=0
            while not terminal:
                if render:   env.render()
                action = agent.act(state)
                nextState, reward, terminal, info = env.step(action)
                nextState = convertTorch(nextState)
                detectorInput = {"S0": state, "S1": nextState, "A": action}
                x, n_x = preprocess(detectorInput)
                envPred = detector.detect(x)
                predicteds[(envPred, envNumber)] += 1
                print("Env: {} Episode: {}/{} StateNo: {}".format(envNumber, e, episodes, i))
                state = nextState
                i=i+1
    return predicteds






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



def trainDetectorDistro(detector, inputDict, envLabel):
    x, n_x = preprocess(inputDict)
    detector.trainDistro(x, envLabel)



def trainDetector(detector, inputDict, envLabel):
    x, n_x = preprocess(inputDict)
    return detector.trainStep(x, n_x, envLabel)



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
    for taskNum in range(len(taskList)):
        print("Task " + str(taskNum) + ":  |", end = '')
        for predNum in range(len(taskList)):
            print("{:8.3f}".format(cm[(predNum, taskNum)]), end = ' ')
        print("|")
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()



def main():
    print("Starting.")
    envNameList = ["breakout", "pong", "space_invaders", "ms_pacman", "assault", "asteroids", "boxing", "phoenix", "alien"]
    atariGames = MultiEnv(envNameList)
    agent = RandomAgent(atariGames.actSpace)
    gen = AnomalyDetectorGenerator(DEVICE, NN_SIZE, H_DIM, LATENT_SIZE)
    detector = TaskDetector(gen, "./%s/" % MODELS_DIR)
    if TRAINING_ON:
        for i, env in enumerate(atariGames.getEnvList()):
            detector.addTask(i)
            train(agent, detector, env, i, TRAIN_EPS, DISTRO_TRAIN_EPS)
            detector.expelDetector(i)
        print("Training complete.")
    else:
        detector.loadAll(envNameList)
        print("Loaded envs %s." % str(envNameList))
    print("Testing with normalize.")
    cm = defaultdict(lambda: 0.0)
    for i, env in enumerate(atariGames.getEnvList()):
        predicteds = test(agent, detector, env, i, TEST_EPS)
        for envNum in range(len(envNameList)):
            for predNum in range(len(envNameList)):
                cm[(predNum, envNum)] += predicteds[(predNum, envNum)]
    print("Testing complete.")
    print()
    printCM(envNameList, cm)
    print()
    print("Testing without normalize.")
    detector.setNormalize(on = False)
    cm = defaultdict(lambda: 0.0)
    for i, env in enumerate(atariGames.getEnvList()):
        predicteds = test(agent, detector, env, i, TEST_EPS)
        for envNum in range(len(envNameList)):
            for predNum in range(len(envNameList)):
                cm[(predNum, envNum)] += predicteds[(predNum, envNum)]
    print("Testing complete.")
    print()
    printCM(envNameList, cm)
    print()
    print("Done.")




if __name__ == '__main__':
    main()

#===============================================================================
