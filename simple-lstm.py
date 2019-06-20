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
AUX_COLUMNS = ['target','severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

#------------------------- util for global objective -----------------------#
def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
  with tf.name_scope(
      name,
      'weighted_logistic_loss',
      [logits, labels, positive_weights, negative_weights]) as name:
    labels, logits, positive_weights, negative_weights = prepare_loss_args(
        labels, logits, positive_weights, negative_weights)

    softplus_term = tf.add(tf.maximum(-logits, 0.0),
                           tf.log(1.0 + tf.exp(-tf.abs(logits))))
    weight_dependent_factor = (
        negative_weights + (positive_weights - negative_weights) * labels)
    return (negative_weights * (logits - labels * logits) +
            weight_dependent_factor * softplus_term)


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0,
                        name=None):
  with tf.name_scope(
      name, 'weighted_hinge_loss',
      [logits, labels, positive_weights, negative_weights]) as name:
    labels, logits, positive_weights, negative_weights = prepare_loss_args(
        labels, logits, positive_weights, negative_weights)

    positives_term = positive_weights * labels * tf.maximum(1.0 - logits, 0)
    negatives_term = (negative_weights * (1.0 - labels)
                      * tf.maximum(1.0 + logits, 0))
    return positives_term + negatives_term


def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.0,
                            negative_weights=1.0,
                            name=None):
  with tf.name_scope(
      name, 'weighted_loss',
      [logits, labels, surrogate_type, positive_weights,
       negative_weights]) as name:
    if surrogate_type == 'xent':
      return weighted_sigmoid_cross_entropy_with_logits(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights,
          name=name)
    elif surrogate_type == 'hinge':
      return weighted_hinge_loss(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights,
          name=name)
    raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def expand_outer(tensor, rank):
  if tensor.get_shape().ndims is None:
    raise ValueError('tensor dimension must be known.')
  if len(tensor.get_shape()) > rank:
    raise ValueError(
        '`rank` must be at least the current tensor dimension: (%s vs %s).' %
        (rank, len(tensor.get_shape())))
  while len(tensor.get_shape()) < rank:
    tensor = tf.expand_dims(tensor, 0)
  return tensor


def build_label_priors(labels,
                       weights=None,
                       positive_pseudocount=1.0,
                       negative_pseudocount=1.0,
                       variables_collections=None):
  dtype = labels.dtype.base_dtype
  num_labels = get_num_labels(labels)

  if weights is None:
    weights = tf.ones_like(labels)

  # We disable partitioning while constructing dual variables because they will
  # be updated with assign, which is not available for partitioned variables.
  partitioner = tf.get_variable_scope().partitioner
  try:
    tf.get_variable_scope().set_partitioner(None)
    # Create variable and update op for weighted label counts.
    weighted_label_counts = tf.contrib.framework.model_variable(
        name='weighted_label_counts',
        shape=[num_labels],
        dtype=dtype,
        initializer=tf.constant_initializer(
            [positive_pseudocount] * num_labels, dtype=dtype),
        collections=variables_collections,
        trainable=False)
    weighted_label_counts_update = weighted_label_counts.assign_add(
        tf.reduce_sum(weights * labels, 0))

    # Create variable and update op for the sum of the weights.
    weight_sum = tf.contrib.framework.model_variable(
        name='weight_sum',
        shape=[num_labels],
        dtype=dtype,
        initializer=tf.constant_initializer(
            [positive_pseudocount + negative_pseudocount] * num_labels,
            dtype=dtype),
        collections=variables_collections,
        trainable=False)
    weight_sum_update = weight_sum.assign_add(tf.reduce_sum(weights, 0))

  finally:
    tf.get_variable_scope().set_partitioner(partitioner)

  label_priors = tf.div(
      weighted_label_counts_update,
      weight_sum_update)
  return label_priors


def convert_and_cast(value, name, dtype):
  return tf.cast(tf.convert_to_tensor(value, name=name), dtype=dtype)


def prepare_loss_args(labels, logits, positive_weights, negative_weights):
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = convert_and_cast(labels, 'labels', logits.dtype)
  if len(labels.get_shape()) == 2 and len(logits.get_shape()) == 3:
    labels = tf.expand_dims(labels, [2])

  positive_weights = convert_and_cast(positive_weights, 'positive_weights',
                                      logits.dtype)
  positive_weights = expand_outer(positive_weights, logits.get_shape().ndims)
  negative_weights = convert_and_cast(negative_weights, 'negative_weights',
                                      logits.dtype)
  negative_weights = expand_outer(negative_weights, logits.get_shape().ndims)
  return labels, logits, positive_weights, negative_weights


def get_num_labels(labels_or_logits):
  """Returns the number of labels inferred from labels_or_logits."""
  if labels_or_logits.get_shape().ndims <= 1:
    return 1
  return labels_or_logits.get_shape()[1].value




# ------------------------------- customized differentiable AUC calculation --------------#
def _prepare_labels_logits_weights(labels, logits, weights):
  # Convert `labels` and `logits` to Tensors and standardize dtypes.
  logits = tf.convert_to_tensor(logits, name='logits')
  labels = convert_and_cast(labels, 'labels', logits.dtype.base_dtype)
  weights = convert_and_cast(weights, 'weights', logits.dtype.base_dtype)

  try:
    labels.get_shape().merge_with(logits.get_shape())
  except ValueError:
    raise ValueError('logits and labels must have the same shape (%s vs %s)' %
                     (logits.get_shape(), labels.get_shape()))

  original_shape = labels.get_shape().as_list()
  if labels.get_shape().ndims > 0:
    original_shape[0] = -1
  if labels.get_shape().ndims <= 1:
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(logits, [-1, 1])

  if weights.get_shape().ndims == 1:
    # Weights has shape [batch_size]. Reshape to [batch_size, 1].
    weights = tf.reshape(weights, [-1, 1])
  if weights.get_shape().ndims == 0:
    # Weights is a scalar. Change shape of weights to match logits.
    weights *= tf.ones_like(logits)

  return labels, logits, weights, original_shape

def roc_auc_loss(
    labels,
    logits,
    weights=1.0,
    surrogate_type='xent',
    scope=None):
  with tf.name_scope(scope, 'roc_auc', [labels, logits, weights]):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(
        labels, logits, weights)

    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = weighted_surrogate_loss(
        labels=tf.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
#    loss = tf.reshape(loss, original_shape)
    return tf.reshape(tf.reduce_sum(loss),[-1])

# ---------------------- masks is a 2D python list that contains all mask including (G+ and G-) (G+ and B-) (G- and B+) ---#
# attention: I am training the model so that y_pred = logit(probability)
def custom_loss(splitted_masks):
    def AUC(y_true,y_pred,mask):
        return roc_auc_loss(y_true,y_pred,weights=mask)

    def loss(y_true,y_pred):
#        AUC_all = roc_auc_loss(y_true,y_pred)
        AUC_id = roc_auc_loss(y_true,y_pred,splitted_masks[0])
#        return tf.reduce_mean(AUCs)*3+AUC_all
#        return tf.add(AUC_all,AUC_id)
        return AUC_id
    return loss
# ---------------------------------------------------------------------------------------------------------------------------#

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
    

def build_model(embedding_matrix, aux_target):
    words = Input(shape=(None,))
    masks = Input(shape=(None,))
    splitted_masks = tf.split(masks,len(IDENTITY_COLUMNS)*3,axis=1)
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
#    result = Dense(1, activation='sigmoid')(hidden)
#    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    if aux_target==0:
        result = Dense(1, activation=None)(hidden)
        model = Model(inputs=[words,masks], outputs=result)
        model.compile(loss=custom_loss(splitted_masks), optimizer='adam')
    else:
        result = Dense(1, activation='sigmoid')(hidden)
        model = Model(inputs=[words,masks], outputs=result)
        model.compile(loss='mean_squared_error', optimizer='adam')

    return model
    
def create_mask(df):
    mask_tot = []
    for identity in IDENTITY_COLUMNS:
        mask0 = df[identity]
        mask1 = (df[identity] & (df['target']<0)) | (~df[identity] & (df['target']>0))
        mask2 = (~df[identity] & (df['target']<0)) | (df[identity] & (df['target']>0))
        print(identity,": number of people ",mask0.sum(),mask1.sum(),mask2.sum())
        masks_tot.append(mask0.values*1.0)
        masks_tot.append(mask1.values*1.0)
        masks_tot.append(mask2.values*1.0)
    masks_tot = np.array(masks_tot).T    
    print("masks_tot shape",masks_tot.shape)
    return mask_tot

train = pd.read_csv('./train_preprocessed.csv')
for column in IDENTITY_COLUMNS:
    train[column] = np.where(train[column] >= 0.5, True, False)
train['target'] = np.where(train[column] >= 0.5,1.0,-1.0)
#Seperate it into train and validation set

end = 64000
split = int(end*0.8)
validate_df = train.iloc[split:end]
train_df = train.iloc[:split]
validate_mask = masks_tot[split:end]
train_mask = masks_tot[:split]
print("validate_df shape",validate_df.shape)
print("train_df shape",train_df.shape)
test_df = pd.read_csv('./test_preprocessed.csv')
print("test and train data imported.")

x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
x_validate = validate_df[TEXT_COLUMN].astype(str)
y_validate = validate_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values
y_aux_validate = validate_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)


train_mask = create_mask(train_df)
validate_mask = create_mask(validate_df)
test_mask = np.array((x_test.shape[0],len(IDENTITY_COLUMNS)*3))

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

predictions = {}
validations = {}
for model_idx in range(NUM_MODELS):
    checkpoint_predictions = []
    checkpoint_validations = []
    weights = []
    model = build_model(embedding_matrix, model_idx)
    for global_epoch in range(EPOCHS):
        model.fit(
            [x_train,masks_tot],
            y_aux_train[:,model_idx],
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
        checkpoint_predictions.append(model.predict([x_test,test_mask], batch_size=2048).flatten())
        checkpoint_validations.append(model.predict([x_validate,validate_mask], batch_size=2048).flatten())
        weights.append(2 ** global_epoch)
    predictions[AUX_COLUMNS[model_idx]] = np.average(checkpoint_predictions, weights=weights, axis=0)
    validations[AUX_COLUMNS[model_idx]] = np.average(checkpoint_validations, weights=weights, axis=0)
"""
submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': scipy.special.expit(predictions)
})
submission.to_csv('submission_custom.csv', index=False)
"""
validations['id'] = validate_df.id
validations['target'] = scipy.special.expit(validations['target'])
submission = pd.DataFrame.from_dict(validations)
submission.to_csv('validation_custom.csv', index=False)
