from __future__ import print_function

import json

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.utils import plot_model

batch_size = 64
epochs = 10000
latent_dim = 256
max_num_samples = 20


# 初始化文件中的数据
def init_dataset(num_samples):
    data_path = './chat.txt'
    # 对数据进行向量化处理
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[:min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # 把\t作为开始符号，把\n作为结束符号
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    # 建立字符集索引，把句子向量化
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])

    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    # encoder输入句子的向量
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    # decoder输入句子的向量
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    # decoder目标句子的向量
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data 比 decoder_input_data 要提前一步
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data 要提前一步，并且不会包含起始字符
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    return {
        'encoder_input_data': encoder_input_data,
        'decoder_input_data': decoder_input_data,
        'decoder_target_data': decoder_target_data,
        'num_encoder_tokens': num_encoder_tokens,
        'num_decoder_tokens': num_decoder_tokens,
        'input_token_index': input_token_index,
        'target_token_index': target_token_index,
        'max_encoder_seq_length': max_encoder_seq_length,
        'max_decoder_seq_length': max_decoder_seq_length,
    }


dataset = init_dataset(max_num_samples)
encoder_input_data = dataset['encoder_input_data']
decoder_input_data = dataset['decoder_input_data']
decoder_target_data = dataset['decoder_target_data']
num_encoder_tokens = dataset['num_encoder_tokens']
num_decoder_tokens = dataset['num_decoder_tokens']

'''建立encoder网络，引入Input层作为LSTM网络输入'''
# 定义输入数据并处理
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# 去掉'encoder_outputs' 只保留状态
encoder_states = [state_h, state_c]
# 设置decoder, 使用'encoder_states'作为初始状态
'''建立decode网络'''
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# 设置decoder返回完整序列
# 与此同时，decoder也需要把内部状态返回
# 我们不在训练时会使用内部状态，但在预测时会使用
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
'''拼接网络、完成模型并进行训练'''
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs,
          validation_split=0.2)

plot_model(model, to_file='s2s_1_model.png', show_shapes=True)

model.save('s2s_1.h5')  # 模型权重
with open('s2s_1.json', 'w', encoding='utf8') as f:
    f.write(model.to_json(indent=4))

config = {
    "latent_dim": 256,
    "max_num_samples": max_num_samples,
    "num_encoder_tokens": num_encoder_tokens,
    "num_decoder_tokens": num_decoder_tokens,
    "input_token_index": dataset['input_token_index'],
    "target_token_index": dataset['target_token_index'],
    "max_encoder_seq_length": dataset['max_encoder_seq_length'],
    "max_decoder_seq_length": dataset['max_decoder_seq_length'],
}

with open('s2s_1_config.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(config))
