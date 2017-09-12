import os
import sys
import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report
import utils
from shallow_neural_networks import ShallowNeuralNetwork
import gensim
import gensim.models.doc2vec as doc2vec
from gensim.models import Doc2Vec
import sys
assert gensim.models.doc2vec.FAST_VERSION > -1 # "this will be sloww otherwise"
from shallow_neural_networks import TfShallowNeuralNetwork

doc2vecContent = {}

# glove_home = "glove.6B"

pos = "positive_examples_clean.txt"
neg = "n_gram_neg_data.txt"
train_size = .9

random.seed()

vocab = {}
splits = {'test': {1.0:[], -1.0: []}, 'train': {1.0:[], -1.0:[]}}

count = 0

#The positive file is split
with open(pos) as posFile:
    posContent = posFile.readlines()
    for line in posContent[000:5000]:
        split = line.split('?')
        split[0] = split[0].strip()
        split[1] = split[1].strip()
        vocab[split[0]] = count
        count +=1
        vocab[split[1]] = count
        count +=1
        newObs = [split[0],split[1]]
        if random.random() < train_size:
            splits['train'][1.0].append(newObs)
        else:
            splits['test'][1.0].append(newObs)

#The negative file is split
with open(neg) as negFile:
    negContent = negFile.readlines()
    for line in negContent[000:5000]:
        split = line.split('|')
        split[0] = split[0].strip()
        split[1] = split[1].strip()
        vocab[split[0]] =  count
        count +=1
        vocab[split[1]] =  count
        count +=1
        newObs = [split[0],split[1]]
        if random.random() < train_size:
            splits['train'][-1.0].append(newObs)
        else:
            splits['test'][-1.0].append(newObs)

dataset = (vocab, splits)

Sim_Pos = 1.0    # Left document 
Dif_Neg = -1.0 # Right document

print "Negative length is: ", len(splits['train'][-1.0]) + len(splits['test'][-1.0])
print "Positive length is: ", len(splits['train'][1.0]) + len(splits['test'][1.0])

# glove50_src = os.path.join(glove_home, 'glove.6B.50d.txt')


# GLOVE50 = utils.glove2dict(glove50_src)

# def glove50vec(w):    
#     """Return `w`'s GloVe representation if available, else return 
#     a random vector."""
#     return GLOVE50.get(w, randvec(w, n=50))

def vec_concatenate(u, v):
    """Concatenate np.array instances `u` and `v` into a new np.array"""
    return np.concatenate((u,v, u-v)) #np.concatenate((u, v, u-v)) #u-v ???

def randvec(w, n=50, lower=-0.5, upper=0.5):
    """Returns a random vector of length `n`. `w` is ignored."""
    return np.array([random.uniform(lower, upper) for i in range(n)])

with open("parsed_data.txt", 'w') as c: 
    count = 0   
    for doc in vocab: 
        c.write(doc + "\n")
        doc2vecContent[doc] = count
        count += 1


sentences = doc2vec.TaggedLineDocument("parsed_data.txt")
model = Doc2Vec(sentences, size=100, window=8, min_count=5, workers=4, iter=20)
model.save("model_name")

def get_vec_for_sentence(sentence):
    if sentence not in doc2vecContent:
        return "Error, sentence not found"
    return(model.docvecs[doc2vecContent[sentence]])

def build_dataset(
        dataset, 
        vector_func,
        vector_combo_func=vec_concatenate): 
    """
    Parameters
    ----------    
    dataset
        The dataset we are analyzing.
    
    vector_func : ()
        Function mapping docs to vector representations
        
    vector_combo_func :
        Function for combining two vectors into a new vector.
        
    Returns
    -------
    dataset : defaultdict
        A map from split names ("train", "test", "disjoint_vocab_test")
        into data instances:
        
        {'train': [(vec, [cls]), (vec, [cls]), ...],
         'test':  [(vec, [cls]), (vec, [cls]), ...],
         'disjoint_vocab_test': [(vec, [cls]), (vec, [cls]), ...]}
    
    """
    # Load in the dataset:
    vocab, splits = dataset
    vectors = {w: vector_func(w) for w in vocab.keys()}
    # Dataset in the format required by the neural network:
    dataset = defaultdict(list)
    for split, data in splits.items():
        for clsname, word_pairs in data.items():
            for w1, w2 in word_pairs:
                item = [vector_combo_func(vectors[w1], vectors[w2]), np.array([clsname])]
                dataset[split].append(item)
    return dataset



def experiment(dataset, network):    
    """
    Parameters
    ----------    
    dataset : dict
        With keys 'train' and 'test', each with values that are lists of vector pairs.
        The expectation is that this was created by `build_dataset`.
    
    network
        This will be `ShallowNeuralNetwork,` but it could be any function that can train and    
        evaluate on `dataset`.
    
    Prints
    ------
    To standard ouput
        An sklearn classification report for all three splits.
    """  
    # Train the network:
    network.fit(dataset['train'])
    for typ in ('train', 'test'):
        data = dataset[typ]
        predictions = []
        cats = []
        for ex, cat in data:            
            # The raw prediction is a singleton list containing a float,
            # either -1 or 1. We want only its contents:
            prediction = network.predict(ex)[0]
            # Categorize the prediction for accuracy comparison:
            prediction = Dif_Neg if prediction <= 0.0 else Sim_Pos            
            predictions.append(prediction)
            # Store the gold label for the classification report:
            cats.append(cat[0])
        # Report:
        print("="*70)
        print(typ)
        print(classification_report(cats, predictions, target_names=['Dif_Neg', 'Sim_Pos']))

baseline_dataset = build_dataset(dataset, vector_func=get_vec_for_sentence, vector_combo_func=vec_concatenate)

baseline_network = ShallowNeuralNetwork(hidden_dim=100, eta=0.05)

experiment(baseline_dataset, baseline_network)

# try:
#     import tensorflow    
# except:
#     print("Warning: TensorFlow is not installed, so you won't be able to use `TfShallowNeuralNetwork`.")

# if 'tensorflow' in sys.modules:
    
#     from shallow_neural_networks import MYTfShallowNeuralNetwork
    
#     baseline_tfnetwork = MYTfShallowNeuralNetwork()
    
#     experiment(baseline_dataset, baseline_tfnetwork)
