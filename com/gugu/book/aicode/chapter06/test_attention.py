from random import randint

from keras import Input, Model
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.layers import LSTM, Flatten, Activation, Permute, multiply
from keras.layers import Dense
from keras.layers import RepeatVector


# 生成一组随机数据
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# 进行one-hot encoding
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# 对编码后的数据进行one-hot decoding
def one_hot_decode(encode_seq):
    return [argmax(vector) for vector in encode_seq]


def get_pair(n_in, n_out, cardinality):
    # 生成随机数
    sequence_in = generate_sequence(n_in, cardinality)

    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    #     one-hot encoding
    X = one_hot_encode(sequence_in, cardinality)
    y = one_hot_encode(sequence_out, cardinality)

    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


def attention_model(n_timesteps_in, n_features):
    units = 50
    inputs = Input(shape=(n_timesteps_in, n_features))

    encoder = LSTM(units, return_sequences=True, return_state=True)
    encoder_outputs, encoder_states, _ = encoder(inputs)

    a = Dense(1, activation='tanh', bias_initializer='zeros')(encoder_outputs)
    a = Flatten()(a)
    annotation = Activation('softmax')(a)
    annotation = RepeatVector(units)(annotation)
    annotation = Permute((2, 1))(annotation)

    context = multiply([encoder_outputs, annotation])
    output = Dense(n_features, activation='softmax', name='final_dense')(context)

    model = Model([inputs], output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
    for epoch in range(5000):
        X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        model.fit(X, y, epochs=1, verbose=0)

    total, corrent = 100, 0
    for _ in range(total):
        X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        yhat = model.predict(X, verbose=0)
        result = one_hot_decode(yhat[0])
        expected = one_hot_decode(y[0])
        if array_equal(expected, result):
            corrent += 1
    return float(corrent) / float(total) * 100.0


n_features = 50
n_timesteps_in = 6
n_timesteps_out = 3
n_repeats = 5

for _ in range(n_repeats):
    model = attention_model(n_timesteps_in, n_features)
    accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
    print(accuracy)
