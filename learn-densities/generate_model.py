import numpy as np
import seaborn as sns
# from sklearn.linear_model import LogisticRegression

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.layers import Embedding, Flatten, Merge
from keras.layers import MaxPooling3D, Conv3D



def generate_model(data):
    n_row, n_col = data.shape
    inputs = Input(shape=(data.shape[1],), dtype='float32', name='spacial_data')
    
    deep_node1 = Dense(units=80, activation='relu', name='deep1')(inputs)
    dropout1 = Dropout(rate=0.2, name='drop1')(deep_node1)

    deep_node2 = Dense(units=50, activation='relu', name='deep2')(dropout1)
    dropout2 = Dropout(rate=0.2, name='drop2')(deep_node2)

    deep_node3 = Dense(units=40, activation='relu', name='deep3')(dropout2)
    dropout3 = Dropout(rate=0.2, name='drop3')(deep_node3)

    deep_node4 = Dense(units=1, activation='relu', name='deep4')(dropout3)

    model = Model(inputs=inputs, outputs=deep_node4)

    return model


def train_model(model, X, y):
      model.compile(optimizer='sgd',
                loss='mean_squared_error',
                metrics=['acc'])

      model.reset_states()

      #training and validating on the same data for now
      history = model.fit(
          X, y, epochs=2, verbose=1,
          validation_data=(X,y))
