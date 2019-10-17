# Word2Mat - A New Word Representation

In this project, we extend the idea of representing words as vectors like Word2Vec to represent them as matrices. The code is based on the Tensorflow Word2Vec tutorial found [Here](https://www.github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
 

## File Description

The project contains the following files:

  - `word2mat.py`: 
    Contains the code for word2mat (documented). 
  - `word2vec_old.py`:
    Original word2vec based on the above Git repo  
  - `word2vec_poc_nce_train.py`:
    Trains the NCE layer using existing word2vec embeddings
  - `wor2vec_prediction_evaluation.py`:
    Calculates word2vec accuracy using existing word2vec embeddings and trained NCE layer weights and biases



## How To Run

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Word2Mat

* ```python word2mat.py | tee outputfile.txt```
    This will run the code which downloads `text8` data and runs the word2mat model. The output will be printed to standard output as well as in the file `outputfile.txt`



### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Word2Vec

* ```python word2vec_old.py | tee outputfile.txt```
    This will also download `text8` data and runs the word2vec model.


### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Word2Vec Prediction Accuracy Evaluation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To validate the word2vec model using on the word prediction task, do the following: 

* `python word2vec_old.py`
    After training, the embeddings will be saved as a `numpy` object named `w2v_embeddings.npy`
* `python word2vec_poc_nce_train.py`
    After NCE layer training, the NCE weights and biases will be saved in `numpy` objects named `w2v_poc_nce_weights.npy` and `w2v_poc_nce_bias.npy`
* ```python wor2vec_prediction_evaluation.py```
    This will print the accuracy of the word2vec model using the trained NCE layer.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Note:** If you changed the hyper parameters in `word2vec_old.py`, make sure to change them in  `word2vec_poc_nce_train.py` and `wor2vec_prediction_evaluation.py` as well.

## Credits
This project was conducted by the following students as a part of the Machine Learning Seminar in Saarland University, in Summer Semester 2018:
Ayan Majumdar, Ehtisham Ali, Mossad Helali, Shahzain Mehboob.
The project was supervised by Prof. Dietrich Klakow.
