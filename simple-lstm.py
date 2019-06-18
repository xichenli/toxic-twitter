import numpy as np
import pandas as pd
import sys,time
sys.path.append("/mnt/home/axichen/python_package")
import scipy
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import loss_layers

EMBEDDING_FILES = [
    './raw_data/crawl-300d-2M.vec',
    './raw_data/glove.840B.300d.txt'
]
NUM_MODELS = 1
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
def custom_loss(masks):
    def AUC(y_true,y_logit,mask):
        indices = tf.Variable(mask,dtype=tf.int64)
        y_true_selected = tf.gather(y_true,indices)
        y_logit_selected = tf.gather(y_logit,indices)
        return loss_layers.roc_auc_loss(y_true_selected,y_logit_selected)

    def loss(y_true,y_pred):
        y_logit = - tf.log(1. / y_pred - 1.)
        AUC_all = loss_layers.roc_auc_loss(y_true,y_logit)
        AUC_id = AUC(y_true,y_logit,masks[0])
#        return tf.reduce_mean(AUCs)*3+AUC_all
        return tf.add(AUC_all,AUC_id)

    return loss

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix
    

def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=result)
    model.compile(loss=custom_loss(masks), optimizer='adam')
#    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
    

train = pd.read_csv('./train_preprocessed.csv')
for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train[column] = np.where(train[column] >= 0.5, True, False)
#Seperate it into train and validation set

end = int(train.shape[0]/1)
split = int(end*0.8)
validate_df = train.iloc[split:end]
train_df = train.iloc[:split]
masks = []
for identity in IDENTITY_COLUMNS:
    mask0 = train_df[identity]
    mask1 = (train_df[identity] & ~train_df['target']) | (~train_df[identity] & train_df['target'])
    mask2 = (~train_df[identity] & ~train_df['target']) | (train_df[identity] & train_df['target'])
    masks.append(train_df.index[mask0].values)
    masks.append(train_df.index[mask1].values)
    masks.append(train_df.index[mask2].values)
print("train_df index",(train_df.index.values)[:100])
print(masks[0][:100])

print("validate_df shape",validate_df.shape)
print("train_df shape",train_df.shape)
test_df = pd.read_csv('./test_preprocessed.csv')
print("test and train data imported.")

x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
x_validate = validate_df[TEXT_COLUMN].astype(str)
y_validate = validate_df[TARGET_COLUMN].values
#y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)


tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train)+list(x_validate)+list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_validate = tokenizer.texts_to_sequences(x_validate)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_validate = sequence.pad_sequences(x_validate, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
print("x_train shape",x_train.shape)
print("x_validate shape",x_validate.shape)
print("test and train data tokenized and padded.")
#sample_weights = np.ones(len(x_train), dtype=np.float32)
#sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
#sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
#sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
#sample_weights /= sample_weights.mean()
#print("test and train data weighted.")
embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

checkpoint_predictions = []
checkpoint_validations = []
weights = []

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, 1)
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            validation_data=None,
            validation_steps=None,
#            validation_data=(x_validate,y_validate),
#            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
            callbacks=[
                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))
            ]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048).flatten())
        checkpoint_validations.append(model.predict(x_validate, batch_size=2048).flatten())
        weights.append(2 ** global_epoch)
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
validations = np.average(checkpoint_validations, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission_custom.csv', index=False)

submission = pd.DataFrame.from_dict({
    'id': validate_df.id,
    'validation': validations
})
submission.to_csv('validation_custom.csv', index=False)
