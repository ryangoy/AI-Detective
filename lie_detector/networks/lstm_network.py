'''Base LSTM model.
Can receive raw input if face net model is given or can receive features extracted with a CNN.
'''


from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, Dropout
from tensorflow.keras import backend as K
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras import layers



# if face_net_model is given, input of shape (batches, frames, x, y, 1)
# if face_net_model is none, input of shape (batches, frames, features)
def LSTM(frames=64, face_net_model=None, end2end="False", lstm_units=64, dense_units=32, weights=None, input_shape=None, dropout=0.5, learning_rate=None):

    img_input = Input(shape=[frames,]+input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = img_input
    if end2end is not "False" and face_net_model is not None:
    	x = TimeDistributed(face_net_model, input_shape=(frames, face_net_model.output_shape[1]))(x)

    x = layers.LSTM(units=lstm_units, return_sequences=False, dropout=dropout)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(1, activation='sigmoid')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = img_input

    model = Model(inputs, x, name='lie_detector_lstm')

    # if weights is not None:
    #     model.load_weights(weights)

    return model