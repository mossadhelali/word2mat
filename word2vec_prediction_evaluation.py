# Step 4: Build and train a skip-gram model.
# ***************************************************************************************** #

batch_size = 128
embedding_size = 200  # Dimension of the embedding vector.
skip_window = 5  # How many words to consider left and right.
num_skips = 8  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.
num_steps = 200501
learning_rate = 0.05

# ***************************************************************************************** #


###############
#os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7' #use GPU with ID=0
config1 = tf.ConfigProto()
config1.allow_soft_placement=True
config1.gpu_options.allocator_type = 'BFC'
config1.gpu_options.per_process_gpu_memory_fraction = 0.95 # maximun alloc gpu50% of MEM
config1.gpu_options.allow_growth = True #allocate dynamically
################






# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'):
    #train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs0 = tf.placeholder(tf.int32, shape=[batch_size])
    train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
 
  loaded_embeddings = np.load('w2v_embeddings.npy')
  loaded_nce_weights = np.load('w2v_nce_weights.npy')
  loaded_nce_bias = np.load('w2v_nce_bias.npy')
    
  print("loaded embeddings of shape: ", np.shape(loaded_embeddings))
  print("loaded nce weights of shape: ", np.shape(loaded_nce_weights))
  print("loaded nce bias of shape: ", np.shape(loaded_nce_bias))
  # Ops and variables pinned to the CPU because of missing GPU implementation
  #with tf.device('/cpu:0'):
  # Look up embeddings for inputs.
  with tf.name_scope('embeddings'):
    #embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embeddings = tf.constant(loaded_embeddings)
    #embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # TODO: match to poc input
    embed0 = tf.nn.embedding_lookup(embeddings, train_inputs0)
    embed1 = tf.nn.embedding_lookup(embeddings, train_inputs1)
    embed = embed0 + embed1  
        
  # Construct the variables for the NCE loss
  with tf.name_scope('weights'):
    nce_weights = tf.constant(loaded_nce_weights)
  with tf.name_scope('biases'):
    nce_biases = tf.constant(loaded_nce_bias)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,   # TODO: check proper input
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  with tf.name_scope('validation_loss'):

    #logits = tf.matmul(validation_embed, tf.transpose(nce_weights))
    logits = tf.matmul(embed, tf.transpose(nce_weights))
    logits = tf.nn.bias_add(logits, nce_biases)
    labels_one_hot = tf.one_hot(train_labels, vocabulary_size, axis=1)
    labels_one_hot = tf.reshape(labels_one_hot, [batch_size, vocabulary_size])

    validation_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot,logits=logits)
    validation_loss = tf.reduce_sum(validation_loss, axis=1)
    print('validation loss: ', validation_loss)
    #TODO: check if reduce sum or reduce mean
    validation_loss = tf.reduce_mean(validation_loss)
  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  with tf.name_scope('predictions'):
    predictions = tf.nn.sigmoid(logits)
    print(predictions)
    



  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Merge all summaries.
  merged = tf.summary.merge_all()

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.



with tf.Session(config=config1, graph=graph) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')


  # Define metadata variable.
  run_metadata = tf.RunMetadata()
  

  # calculate loss on validation set
  average_validation_loss = 0
  average_accuracy = 0  

  for i in range(int(validation_size/batch_size)):
    validation_batch_inputs0, validation_batch_inputs1, validation_batch_labels, validation_batch_context_dict = generate_validation_batch(batch_size, num_skips,skip_window)

    #validation_batch_inputs0 = [i[0] for i in validation_batch_inputs]
    #validation_batch_inputs1 = [i[1] for i in validation_batch_inputs]
    #validation_feed_dict = {validation_inputs0: validation_batch_inputs0, validation_inputs1: validation_batch_inputs1, validation_labels: validation_batch_labels}
    validation_feed_dict = {train_inputs0: validation_batch_inputs0, train_inputs1: validation_batch_inputs1, train_labels: validation_batch_labels}

    '''
        separate dict into two lists of input words
        get the embeddings for each list
        multiply the embeddings
        flatten the result
        multiply by nce weights
        add nce bias
        run tf.nn.sigmoid
        get argmax (id of the word)
        check if the id is in the context list
        calculate accuracy
        add to avg accuracy
    '''


    validation_loss_value, context_predictions = session.run([validation_loss, predictions], feed_dict = validation_feed_dict, run_metadata=run_metadata)

    average_validation_loss = average_validation_loss + validation_loss_value

    unique_tuple_indices = np.arange(0, batch_size, num_skips)
    unique_predictions = context_predictions[unique_tuple_indices]
    unique_tuples = np.array(list(zip(validation_batch_inputs0, validation_batch_inputs1)))[unique_tuple_indices]

    unique_predictions_argmax = np.argmax(unique_predictions, axis=1)

    correct_prediction_count = 0
    for i in range(len(unique_tuples)):
        unique_tuple = tuple(unique_tuples[i])

        prediction = unique_predictions_argmax[i]
        context_words = validation_batch_context_dict[unique_tuple]
        if(prediction in context_words):
            correct_prediction_count += 1

    average_accuracy += correct_prediction_count/len(unique_tuples)

  average_validation_loss = average_validation_loss / (validation_size / batch_size)
  average_accuracy /= (validation_size / batch_size)

  print("Average Validation Loss of %d batches: %.5f" % (int(validation_size / batch_size), average_validation_loss))
  print("Average Accuracy of %d batches: %.5f" % (int(validation_size / batch_size), average_accuracy))

    
        

  final_embeddings = normalized_embeddings.eval()

  # Write corresponding labels for the embeddings.
  with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')


  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)