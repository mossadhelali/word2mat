# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
from datetime import datetime

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector


"""Basic word2vec example."""


'''
Word2Mat 

This code is built on the Tensorflow word2vec tutorial found at: https://www.github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

The code is written in tensorflow and runs on a signle GPU.
In this work, we operate on bigrams and try to improve the embedding of each word by running the skip-gram model to predict the context.

The general flow of the code is as follows:
1. download text8 dataset and generate all training data (bigrams and labels)
2. train the model on the train data
    a. take each bigram one-hot 
    b. multiply by the projection tensor to get embeddings
    c. multiply the two embeddings
    d. flatten the resulting embedding, feed with the labels to nce_loss 
    c. update the model params, i.e embeddings, nce wights and nce biases
3. evaluate the model on validation data every epoch
    a. calcualte validation loss and prediction accuracy:
        1. take each bigram in the validation set
        2. multiply by the projection tensor to get embeddings
        3. multiply the two embeddings
        4. flatten the resulting embedding, multiply by nce wights and add nce biases
        5. calculate argmax of sigmoid of results to get the loss and prediction of each bigram, compare prediction with the context
    b. calculate the validation loss and prediction accuracy for the flipped bigrams
       we show that when flipping the order of the bigram, the accuracy is less. i.e. word2mat is context sensitive.
    c. calcluate the nearest words for each word in a list of common words (word similarity)
        1. normalize each embedding
        2. calcualte cosine similarity between all embeddings
        3. for each word in the list, print the words which have the closest embeddings
    d. sentence completion of chosen bigrams
        1. similary to train and validation, get the two embeddings
        2. multiply them, flatten the result, multiply by nce weights and biases
        3. calculate top 10 of the sigmoid of the result to get the top 10 predicted words

'''



start_time = datetime.now()
print('started at: ', start_time, '\n')


# Data URL
url = 'http://mattmahoney.net/dc/'


# Hyper parameters
# ***************************************************************************************** #

batch_size = 400                                     # The size of each training & validation batch
embedding_size = 400                                 # Dimension of the embedding vector.
embedding_size_sqrt = int(math.sqrt(embedding_size)) # Dimension of the embedding matrix.
skip_window = 2                                      # How many words to consider left and right.
num_skips = 4                                        # How many words to predict from the context for each bigram
num_sampled = 300                                    # Number of negative examples to sample.
num_steps = 6000001                                  # Number of training batches to generate. num epochs = steps / (train data size/(batch_size / num_skips))
starting_learning_rate = 0.06                        # Starting learning rate, decays with each step
min_learning_rate = 0.0001                           # Min learning rate.

# ***************************************************************************************** #

# GPU configuration
###############
os.environ["CUDA_VISIBLE_DEVICES"] = '0'                    #use GPU with ID=0
config1 = tf.ConfigProto()
config1.allow_soft_placement=True
config1.gpu_options.allocator_type = 'BFC'
config1.gpu_options.per_process_gpu_memory_fraction = 0.45  # maximun alloc gpu 45% of MEM
config1.gpu_options.allow_growth = True                     # allocate dynamically
#config1.log_device_placement=True                          # log which variable is allocated to which device (GPU or CPU)
################


print('\nHyper Parameters:')
print('Num Steps:\t', num_steps)
print('Leanring Rate:\t', starting_learning_rate)
print('Batch Size:\t', batch_size)
print('Embedding Size:\t', embedding_size)
print('Skip Window:\t', skip_window)
print('Num Skips:\t', num_skips)
print('Neg Samples:\t', num_sampled, '\n')



# Checks whether the data is downloaded, if not it downloads from the data url 
# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename



# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data




vocabulary_size = 50000  # number of different words in the vocabulary. The rest will be replaced by UNK

# Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary




validation_size = 1000000      # how many words to use for validation out of all text8 data (16M)


data_index = 0


# generates a single training batch for the skip-gram model.
# each batch consits of a list of bigrams and labels
# the bigrams are put into two lists, batch0 and batch1.
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch0 = np.ndarray(shape=(batch_size), dtype=np.int32)             # a list of the first words in each bigram
  batch1 = np.ndarray(shape=(batch_size), dtype=np.int32)             # a list of the seconds words in each bigram
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  span = 2 * skip_window + 2  # [ skip_window target target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if (w != skip_window and w != (skip_window + 1))]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch0[i * num_skips + j] = buffer[skip_window]
      batch1[i * num_skips + j] = buffer[skip_window + 1]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data) 
  return batch0, batch1, labels


# generate the whole training set to allow for batch normalization.
# Note: this is not implemented in the orignial w2v code.
all_batches0 = []
all_batches1 = []
all_labels = []

# download the data, read it, and build the dataset
filename = maybe_download('text8.zip', 31344016)
vocabulary = read_data(filename)
print('Data size', len(vocabulary))
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

validation_data = data[1 - validation_size:]      # reserve the last part of the data to validation set
data = data[0:len(data) - validation_size + 1]

del vocabulary  # Hint to reduce memory.

num_batches = math.floor(len(data) / (batch_size / num_skips))

print('generating all ', num_batches, ' training batches ...')


# generate all training batches
for i in range(num_batches):
  batch0, batch1, labels = generate_batch(data, batch_size, num_skips, skip_window)
  all_batches0 += batch0.tolist()
  all_batches1 += batch1.tolist()
  all_labels += labels.tolist()

all_batches0 = np.array(all_batches0)
all_batches1 = np.array(all_batches1)
all_labels = np.array(all_labels)

    

num_batches_in_epoch = len(all_batches0) / batch_size
index_seq = list(range(len(all_batches0)))

# supply one training batch and shuffle the order of the data with each epoch
batch_index = -batch_size
def give_batch(current_batch):

    global index_seq
    global batch_index
    
    if(current_batch % num_batches_in_epoch == 0):
        print('One epoch passed, shuffling data ...')
        np.random.shuffle(index_seq)
        batch_index = -batch_size
    
    batch_index += batch_size
    
    return all_batches0[index_seq[batch_index:batch_index+batch_size]], all_batches1[index_seq[batch_index:batch_index+batch_size]], all_labels[index_seq[batch_index:batch_index+batch_size]]
             

validation_index = 0


# same logic as generate_batch, but also returns a list of context words for each bigram to allow for prediction accuracy calculation
# note that the length of context_list is less than (batch0, batch1, labels), due to non-repetition 
def generate_validation_batch(batch_size, num_skips, skip_window):
  global validation_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  context_list = []  

  batch0 = np.ndarray(shape=(batch_size), dtype=np.int32)
  batch1 = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 2  
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if validation_index + span > len(validation_data):
    validation_index = 0
  buffer.extend(validation_data[validation_index:validation_index + span])
  validation_index += span
  
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if (w != skip_window and w != (skip_window + 1))]
    words_to_use = random.sample(context_words, num_skips)
    context_list.append([])
    for j, context_word in enumerate(words_to_use):
      batch0[i * num_skips + j] = buffer[skip_window]
      batch1[i * num_skips + j] = buffer[skip_window + 1]
      labels[i * num_skips + j, 0] = buffer[context_word]
      context_list[i].append(buffer[context_word])      # for each bigram, make a list of the words sampled from the context


    if validation_index == len(validation_data):
      buffer.extend(validation_data[0:span])
      validation_index = span
    else:
      buffer.append(validation_data[validation_index])
      validation_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  validation_index = (validation_index + len(validation_data) - span) % len(validation_data) 
  return batch0, batch1, labels, context_list




# list of words for the word similarity task (3.c)
# rather than selecting from the top 100 words like in the original w2v,
# the words are chosen from different areas of language such as: nubmers, countries, colors, games, etc
validation_word_list = ['five', 'he','computer','france','clinton','company','oxygen','dog','game','tree','rain','book','man','banana','car','football', 'orange', 'green','winter','apple']
validation_word_indices = [dictionary[i] for i in validation_word_list]
valid_size = len(validation_word_list) 
valid_examples = validation_word_indices




graph = tf.Graph()

with graph.as_default():
  with tf.device('/device:GPU:0'):
  # Input data.
    with tf.name_scope('inputs'):
      train_inputs0 = tf.placeholder(tf.int32, shape=[batch_size])  # used for train and validation inputs
      train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size])
      
      test_inputs0 = tf.placeholder(tf.int32, shape=[6])            # used for sentence completion inputs
      test_inputs1 = tf.placeholder(tf.int32, shape=[6])
    
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
      learningRateDecayed = tf.placeholder(tf.float32)              # learning rate is supplied as a placeholder to allow leraning rate decay

    
    # embeddings initialization
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size_sqrt, embedding_size_sqrt], -1.0, 1.0))  
      embed0 = tf.nn.embedding_lookup(embeddings, train_inputs0)                # get the embeddings of the two words
      embed1 = tf.nn.embedding_lookup(embeddings, train_inputs1)    
      embed_intermediate = tf.matmul(embed0, embed1)                            # multiply to get the intermidiate embedding        
      embed = tf.reshape(embed_intermediate, [batch_size, embedding_size])      # flatten the intermidiate embedding
    

      test_embed0 = tf.nn.embedding_lookup(embeddings, test_inputs0)
      test_embed1 = tf.nn.embedding_lookup(embeddings, test_inputs1)
      test_embed_intermediate = tf.matmul(test_embed0, test_embed1)
      test_embed = tf.reshape(test_embed_intermediate, [6, embedding_size])
    
  # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size))) 
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # calculate the nce loss for training batches
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,   
              num_sampled=num_sampled,
              num_classes=vocabulary_size))
  
    # calculate the validation loss 
    # note there is no negative sampling here
    with tf.name_scope('validation_loss'):
  
      logits = tf.matmul(embed, tf.transpose(nce_weights))
      logits = tf.nn.bias_add(logits, nce_biases)
      labels_one_hot = tf.one_hot(train_labels, vocabulary_size)
      labels_one_hot = tf.reshape(labels_one_hot, [batch_size, vocabulary_size])
      validation_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot,logits=logits)
      validation_loss = tf.reduce_sum(validation_loss, axis=1)
      validation_loss = tf.reduce_mean(validation_loss)

    
    test_logits = tf.matmul(test_embed, tf.transpose(nce_weights))
    test_logits = tf.nn.bias_add(test_logits, nce_biases)

    
    with tf.name_scope('predictions'):
      predictions = tf.nn.sigmoid(logits)

    test_predictions = tf.nn.sigmoid(test_logits)  
    
    # momentum optimizer with leraning rate decay
    with tf.name_scope('optimizer'):
      optimizer = tf.train.MomentumOptimizer(learning_rate=learningRateDecayed, momentum=0.9).minimize(loss)








    # Word similarity task, calculate similarity between the embeddings in the list and all other embeddings
    # normalize each matrix and calcluate dot product between them
    norm = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True), 2, keepdims=True))

    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
  
    similarity = tf.tensordot(valid_embeddings, normalized_embeddings, [[1,2], [1,2]])
    
    init = tf.global_variables_initializer()



# Begin training.
with tf.Session(config=config1, graph=graph) as session:


  # initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  learning_rate = starting_learning_rate
  printed_nan_once = False                 
    
  # log of values   
  step_history = []                             # at which step is the evaluation  (x axis)
  loss_value_history = []                       # train loss
  validation_loss_value_history = []            # validation set loss (on the last 1M words of the data)
  accuracy_history = []                         # accuracy of word prediction task on validation set
  reverse_validation_loss_value_history = []    # validation loss when each bigram is flipped
  reverse_accuracy_history = []                 # accuracy of word prediction task on validation set when each bigram is flipped


  # run training for num_steps steps, each step has one batch
  for step in xrange(num_steps):
    batch_inputs0, batch_inputs1, batch_labels = give_batch(step) # get a training batch
    

    # learning rate decay
    # decays with each step
    learning_rate = starting_learning_rate * ( 1 - ((step * 1.0) / num_steps))
    learning_rate = round(learning_rate, 4)
    learning_rate = max(min_learning_rate, learning_rate)
    
    
    feed_dict = {train_inputs0: batch_inputs0, train_inputs1: batch_inputs1, train_labels: batch_labels, learningRateDecayed: learning_rate}
    

    _,  loss_val = session.run([optimizer,loss],feed_dict=feed_dict)
    
    # if loss becomes NaN because of learning rate (or other hyper param combination), print at which step it occured
    if math.isnan(loss_val) and not printed_nan_once:
        print("NaN loss at step: ", step)
        printed_nan_once = True

    average_loss += loss_val

    
    # The average loss is an estimate of the loss over the last 2000 batches.
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000

      print('Average loss at step ', step, ': ', '{0:.5f}'.format(average_loss), '    lr: ', learning_rate)
      average_loss = 0

    # run the word similarity, validation loss, word prediction and sentence completion tasks after each epoch 
    if step % (len(all_batches0) / batch_size) == 0:

      # word similarity taks
      sim = similarity.eval()

      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]

        
        log_str = 'Nearest to %s: ' % valid_word

        
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]     
          log_str = '%s %s,' % (log_str, close_word)

         
            
        print(log_str)
        
      # validation loss and word prediction
      print('Calculating validation loss and accuracy ...')  
      # calculate loss on validation set
      average_validation_loss = 0
      average_accuracy = 0
      reverse_average_validation_loss = 0
      reverse_average_accuracy = 0
    
      '''
      validation is run in batches of the same size as train batches, to be able to reuse the same tf variables and save space
      for each batch, calculate the validation loss and prediction accuracy for the bigrams and the flipped bigrams
      '''
      for i in range(int(validation_size / batch_size)):
        validation_batch_inputs0, validation_batch_inputs1, validation_batch_labels, validation_batch_context_list = generate_validation_batch(batch_size, num_skips,skip_window)
        
        # loss
        validation_feed_dict = {train_inputs0: validation_batch_inputs0, train_inputs1: validation_batch_inputs1, train_labels: validation_batch_labels}     
        validation_loss_value = session.run(validation_loss, feed_dict = validation_feed_dict)
        context_predictions = session.run(predictions, feed_dict = {train_inputs0: validation_batch_inputs0, train_inputs1: validation_batch_inputs1})
        
        average_validation_loss += validation_loss_value
        
        # loss of flipped bigram
        reverse_validation_feed_dict = {train_inputs0: validation_batch_inputs1, train_inputs1: validation_batch_inputs0, train_labels: validation_batch_labels}
        reverse_validation_loss_value = session.run(validation_loss, feed_dict = reverse_validation_feed_dict)
        reverse_context_predictions = session.run(predictions, feed_dict = {train_inputs0: validation_batch_inputs1, train_inputs1: validation_batch_inputs0})
        
        reverse_average_validation_loss += reverse_validation_loss_value
        
        # pick the predictions of each unique bigram in the batches (hint: each batch has each bigram repeated num_skips times)
        unique_tuple_indices = np.arange(0, batch_size, num_skips)
        
        unique_predictions = context_predictions[unique_tuple_indices]
        reverse_unique_predictions = reverse_context_predictions[unique_tuple_indices]
        
        
        unique_predictions_argmax = np.argmax(unique_predictions, axis=1)
        reverse_unique_predictions_argmax = np.argmax(reverse_unique_predictions, axis=1)
        
        assert len(unique_predictions_argmax) == len(validation_batch_context_list) 
        
        correct_prediction_count = 0
        for i in range(len(unique_predictions_argmax)):
            # if the prediction is a word that has occured in the context, it is considered a correct
            if unique_predictions_argmax[i] in validation_batch_context_list[i]:
                correct_prediction_count += 1
                
        reverse_correct_prediction_count = 0
        for i in range(len(unique_predictions_argmax)):
            if reverse_unique_predictions_argmax[i] in validation_batch_context_list[i]:
                reverse_correct_prediction_count += 1
        
        average_accuracy += correct_prediction_count/len(unique_predictions_argmax)
        reverse_average_accuracy += reverse_correct_prediction_count/len(unique_predictions_argmax)
    
      average_validation_loss /= (validation_size / batch_size)
      reverse_average_validation_loss /= (validation_size / batch_size)
        
      average_accuracy /= (validation_size / batch_size)
      reverse_average_accuracy /= (validation_size / batch_size)
      
      
      print(int(validation_size / batch_size), ' validation batches\n')
      print("\nAverage Validation Loss: %.5f" %  average_validation_loss)
      print("Average Accuracy: %.5f\n" % average_accuracy)
    
      print("\nAverage Validation Loss (Reversed): %.5f" %  reverse_average_validation_loss)
      print("Average Accuracy (Reversed): %.5f\n" % reverse_average_accuracy)
        
      step_history.append(step)
      loss_value_history.append(loss_val)
      validation_loss_value_history.append(average_validation_loss)
      accuracy_history.append(average_accuracy)
      reverse_validation_loss_value_history.append(reverse_average_validation_loss)
      reverse_accuracy_history.append(reverse_average_accuracy)
    
    
    
    
      # sentence completion task, bigrams are chosen to show that flipping them will result in different completions
      sentence_words0 = [dictionary[i] for i in ['wall', 'white', 'loud', 'university', 'french', 'deep']]
      sentence_words1 = [dictionary[i] for i in ['street', 'house', 'music', 'of', 'capital', 'learning']]
    
      sentence_feed_dict = {test_inputs0: sentence_words0, test_inputs1: sentence_words1}
      reverse_sentence_feed_dict = {test_inputs0: sentence_words1, test_inputs1: sentence_words0}
      

      sentence_predictions = session.run(test_predictions, feed_dict=sentence_feed_dict)
      reverse_sentence_predictions = session.run(test_predictions, feed_dict=reverse_sentence_feed_dict)
      unique_sentence_predictions = np.argsort(sentence_predictions, axis=1)[:, -10:]                       # pick the top 10 completions
      reverse_unique_sentence_predictions = np.argsort(reverse_sentence_predictions, axis=1)[:, -10:]
        
      print('')
      for i in range(len(sentence_words0)):
        print('prediction of ', reverse_dictionary[sentence_words0[i]] ,'-', reverse_dictionary[sentence_words1[i]], ': ', [reverse_dictionary[j] for j in unique_sentence_predictions[i]])
        print('prediction of ', reverse_dictionary[sentence_words1[i]] ,'-', reverse_dictionary[sentence_words0[i]], ': ', [reverse_dictionary[j] for j in reverse_unique_sentence_predictions[i]], '\n')

    
  # print the logs
  print('graph values:\n')
  print('Steps:\n', step_history, '\n')
  print('Training Loss:\n', loss_value_history, '\n')
  print('Validation Loss:\n', validation_loss_value_history, '\n')
  print('Accuracy:\n', accuracy_history)
  print('Validation Loss (Reversed):\n', reverse_validation_loss_value_history, '\n')
  print('Accuracy (Reversed):\n', reverse_accuracy_history)

  
  
  end_time = datetime.now()
  time_diff = end_time - start_time
  print('Ended at: ', end_time)
  print('Total time: ', time_diff.days, ' days, ', time_diff.seconds//3600, ' hours, ', (time_diff.seconds//60)%60, ' mins')  

