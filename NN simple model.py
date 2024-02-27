# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:33:50 2024

@author: t.breeman
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import pickle
import numpy as np
import sys
import math 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
 
#init and config should be place here, since they are in a different folder
from init import * 
from config import *

#To DO:
    #Evaluatie checken, of dit inderdaad de loss geeft. 
    #Daarnaast kijken of we Adam kunnen configureren op halverings learning rate elke 500 epochs.
    # Input van de sets krijgen.
    # Opslaan van het uiteindelijke model.
    # Nadenken of we Tensorboard willen gebruiken voor inzicht in model.

dim_in = 4
dim_out = 2
#Onderstaande lijnen eventueel gebruiken voor splitsen van de sets. Met verandering van dimensies. 

train, test = train_test_split(newdf, test_size=0.2, random_state=2)
train, val = train_test_split(train, test_size=0.2, random_state=23)

train_y = (np.take(np.array(train),[dim_in,dim_in+3],axis=1))
test_y = (np.take(np.array(test),[dim_in,dim_in+3],axis=1))
val_y = (np.take(np.array(val),[dim_in,dim_in+3],axis=1))


train_x = np.array([v[:dim_in] for v in train])
test_x = np.array([v[:dim_in] for v in test])
val_x = np.array([v[:dim_in] for v in val])


def simple_ANN_model():   
    model = Sequential()
    model.add(Dense(200, activation= 'relu', input_shape=(dim_in,)))
    model.add(Dropout(0))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0))
    model.add(Dense(200, activation='linear'))
    model.add(Dense(dim_out))  # Output layer for the 5 floats
    
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model

model = create_model()
#Verbose even checken wat we willen (hoeveel willen we zien tijdens training, documentatie voor raadplegen)
#Validatie set moet hierin nog eventueel toegevoegd worden.
model.fit(x_train, y_train, batch_size= 1024, epochs = 8000, verbose=2)

# evaluation
loss = model.evaluate(x_test, y_test)
