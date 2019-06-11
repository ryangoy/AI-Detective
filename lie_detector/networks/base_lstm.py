'''Base LSTM model.
Can receive raw input if face net model is given or can receive features extracted with a CNN.
'''


from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from lie_detector.weights import get_weights as utils
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers


# if face_net_model is given, input of shape (batches, frames, x, y, 1)
# if face_net_model is none, input of shape (batches, frames, features)
def LSTM(frames, face_net_model=None, hidden_units=512, weights=None, input_tensor=None, input_shape=None, dropout=0.5):
	if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if face_net_model is not None:
    	x = TimeDistributed(face_net_model, input_shape=(frames, face_net_model.output_shape[1]))(x)

    x = LSTM(units=hidden_units, return_sequences=False, dropout=dropout)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1)(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='lie_detector_lstm')

    # load weights
    if weights == 'vggface':

        weights_path = get_file('senet50_vggface_weights.h5', utils.SENET50_WEIGHTS_NO_TOP_URL,
                                cache_subdir=utils.CACHE_PATH)
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='classifier')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)


    return model