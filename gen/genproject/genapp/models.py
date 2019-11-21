from django.db import models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.iterators import SerialIterator
from chainer import optimizers
from chainer.optimizer_hooks import WeightDecay
from chainer.serializers import save_npz

##ネットワーク決定
class Net(chainer.Chain):

    #n_in:入力層, n_hidden:中間層, n_out:出力層
    def __init__(self, n_in=87, n_hidden=100, n_out=6):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return h