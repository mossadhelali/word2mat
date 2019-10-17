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
"""Basic word2vec example."""

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
import datetime

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()


if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)


url = 'http://mattmahoney.net/dc/'


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


filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))


vocabulary_size = 50000


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



data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch = np.ndarray(shape=(batch_size), dtype=(np.int32, 2))
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  span = 2 * skip_window + 2 
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):

    context_words = [w for w in range(span) if (w != skip_window and w != (skip_window + 1))]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):

      batch[i * num_skips + j] = (buffer[skip_window], buffer[skip_window + 1])
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1

  data_index = (data_index + len(data) - span) % len(data) 
  return batch, labels


batch, labels = generate_batch(batch_size=12, num_skips=2, skip_window=3)

    
    

# Step 4: Build and train a skip-gram model.
# ***************************************************************************************** #

batch_size = 400
embedding_size = 400  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.
num_sampled = 300  # Number of negative examples to sample.
num_steps = 200501
learning_rate = 0.05

# ***************************************************************************************** #


###############
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config1 = tf.ConfigProto()
config1.allow_soft_placement=True
config1.gpu_options.allocator_type = 'BFC'
config1.gpu_options.per_process_gpu_memory_fraction = 0.95 # maximun alloc gpu50% of MEM
config1.gpu_options.allow_growth = True #allocate dynamically
################




validationWordList = ['six','nine','he','computer','france','good','company','bed','cat','game','tree','and','book','man','red','car','football','green','winter','apple']
validationWordIndices = []
for i in range(len(validationWordList)):
    validationWordIndices.append(dictionary[validationWordList[i]])




valid_size = 20  

valid_examples = validationWordIndices

graph = tf.Graph()

with graph.as_default():
  
  loaded_embeddings = np.load('w2v_embeddings.npy')

  # Input data.
  with tf.name_scope('inputs'):

    train_inputs0 = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


  with tf.name_scope('embeddings'):
    embeddings = tf.constant(loaded_embeddings)

    embed0 = tf.nn.embedding_lookup(embeddings, train_inputs0)
    embed1 = tf.nn.embedding_lookup(embeddings, train_inputs1)
    embed = embed0 + embed1  
        

  with tf.name_scope('weights'):
    nce_weights = tf.Variable(
        tf.truncated_normal(
            [vocabulary_size, embedding_size],
            stddev=1.0 / math.sqrt(embedding_size)))
  with tf.name_scope('biases'):
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,   
            num_sampled=num_sampled,
            num_classes=vocabulary_size))


  tf.summary.scalar('loss', loss)

    
    

  with tf.name_scope('optimizer'):

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)






  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)


  merged = tf.summary.merge_all()


  init = tf.global_variables_initializer()


  saver = tf.train.Saver()



with tf.Session(config=config1, graph=graph) as session:

  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)


  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
    
    batch_inputs0 = [i[0] for i in batch_inputs]
    batch_inputs1 = [i[1] for i in batch_inputs]
    

    feed_dict = {train_inputs0: batch_inputs0, train_inputs1: batch_inputs1, train_labels: batch_labels}
    

    run_metadata = tf.RunMetadata()


    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val


    writer.add_summary(summary, step)

    if step == (num_steps - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000

      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0


    if step % 20000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


  nce_weights_to_store = nce_weights.eval()
  nce_bias_to_store = nce_biases.eval()
  
  timestr = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_')


  
  np.save('w2v_poc_nce_weights.npy', nce_weights_to_store)
  np.save('w2v_poc_nce_bias.npy', nce_bias_to_store)
  
  print('saved weights and biases in')
  print('w2v_poc_nce_weights.npy')
  print('w2v_poc_nce_bias.npy')
    

writer.close()

