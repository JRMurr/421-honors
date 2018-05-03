from keras.layers import Embedding, LSTM, Dense, Input
from keras.models import Sequential, Model
from keras import layers
import keras

def create_base_network(max_words, embedding_dim, maxlen, embedding_matrix=None):
    """ Base network to be shared (eq. to feature extraction).
    """
    model = Sequential()
    if embedding_matrix is not None:
        model.add(Embedding(max_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False))
    else:
        model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(LSTM(embedding_dim))
    return model


def build_concate_input(max_words, embedding_dim, maxlen, embedding_matrix=None):
    """Builds the network upto the concatenated layer"""
    base = create_base_network(max_words, embedding_dim, maxlen, embedding_matrix)
    in1 = Input(shape=(maxlen,))
    in2 = Input(shape=(maxlen,))
    encoded_1 = base(in1)
    encoded_2 = base(in2)
    concatenated = layers.concatenate([encoded_1, encoded_2], axis=-1)
    return concatenated, in1, in2


def build_snli(max_words, embedding_dim, maxlen, embedding_matrix=None, model_file='SNLI_best_model_learned_embed.h5'):
    concatenated, in1, in2 = build_concate_input(max_words, embedding_dim, maxlen)
    predictions = Dense(1, activation='sigmoid')(concatenated)
    model = Model(inputs=[in1, in2], outputs=predictions)
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='acc',
            patience=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_file,
            monitor='acc',
            save_best_only=True,
        )
    ]
    return model, callbacks_list