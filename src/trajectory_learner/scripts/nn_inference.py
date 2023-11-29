#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:18:53 2023

@author: sourav
"""


import numpy as np
from physics_guided_nn import PgNN

dataset = np.load("data/dataset.npy", allow_pickle=True)
_dict = dataset.item()
X = np.array([*_dict.keys()])
Y = np.array([*_dict.values()])

model = PgNN()
model.compile(loss= PgNN.Physics_Loss, optimizer='adam')
model.load_weights('model')
# import pdb; pdb.set_trace()
print(model.predict([3.0]))


