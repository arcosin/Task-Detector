
import torch
import torch.nn as nn

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock, ProgDeformConv2DBlock, ProgDeformConv2DBNBlock, ProgConvTranspose2DBNBlock

class ProgVariationalBlock(ProgInertBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module1 = nn.Linear(inSize, outSize)
        self.module2 = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation
        self.gMu = None
        self.gVar = None

    def runBlock(self, x):
        mu = self.module1(x)
        var = self.module2(x)
        return (mu, var)

    def runActivation(self, x):
        self.gMu, self.gVar = x
        return (self.gMu, self.gVar)

    def getData(self):
        data = dict()
        data["type"] = "Variational"
        data["input_sizes"] = [self.inSize]
        data["output_sizes"] = [self.outSize, self.outSize]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)

    def getDistribution(self):
        return (self.gMu, self.gVar)



def reparamaterize(x):
    mu, logVar = x
    std = torch.exp(0.5 * logVar)
    eps = torch.randn_like(std)
    return eps * std + mu



class VariationalAutoEncoderModelGenerator(ProgColumnGenerator):
    def __init__(self, zDim):
        self.ids = 0
        self.lastVarBlock = None
        self.z = zDim

    def generateColumn(self, parentCols, msg = None):
        cols = []
        if msg is None:
            msg = self.__genID()
        # Encoder.
        cols.append(ProgDeformConv2DBNBlock(3, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(32, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(64, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(128, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgDeformConv2DBNBlock(256, 512, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgLambdaBlock(512, 512 * 4, lambda x: torch.flatten(x, start_dim=1)))
        # Latent.
        self.lastVarBlock = ProgVariationalBlock(512 * 4, self.z, len(parentCols))
        cols.append(self.lastVarBlock)
        cols.append(ProgLambdaBlock(self.z, self.z, reparamaterize))
        # Decode
        cols.append(ProgDenseBlock(self.z, 512 * 4, len(parentCols), activation=None))
        cols.append(ProgLambdaBlock(512 * 4, 512, lambda x: x.view(-1, 512, 2, 2)))
        cols.append(ProgConvTranspose2DBNBlock(512, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(256, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(128, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(64, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgConvTranspose2DBNBlock(32, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(ProgDeformConv2DBlock(32, 3, 3, len(parentCols), activation=nn.Tanh(), layerArgs={'padding': 1}))
        return ProgColumn(msg, cols, parentCols = parentCols)

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

    def getLastVarBlock(self):
        return self.lastVarBlock


#===============================================================================
