# Papers
Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks: http://aclweb.org/anthology/D/D15/D15-1181.pdf
    
    -Implementation in torch https://hohocode.github.io/textSimilarityConvNet/


Learning discriminative projections for text similarity measures https://dl.acm.org/citation.cfm?id=2018965 

    - Talks about how TF-IDF measure works well but doesn't handle words that have similar semantic meaning well
    - Learns a projection of the word vector to a low dimmensonal space so the semantic relationships still hold
    - Simple setup, first layer is just the word vec input, 2nd layer is out put. Uses no activation function so it is just a linear projection
    - Find similairty with cosine distance btwn the projected vectors
    - Loss is pairwise loss instead of looking at individual pairs, this makes it better at ranking instead of computing direct similairty 

Learning Text Similarity with Siamese Recurrent Networks http://www.aclweb.org/anthology/W16-16#page=162

    - Uses siamese network structure on variable length strings
    - LSTM 
    - Generates word emdedings then does cos similarity


The Evaluation of Sentence Similarity Measures https://www.researchgate.net/profile/Palakorn_Achananuparp/publication/220802383_The_Evaluation_of_Sentence_Similarity_Measures/links/0deec52cb85c20b04a000000.pdf


Sentence Pair Scoring: Towards Unified Framework for Text Comprehension https://arxiv.org/abs/1603.06127


# Models

https://github.com/dhwajraj/deep-siamese-text-similarity


# Data
https://nlp.stanford.edu/projects/snli/

https://github.com/brmson/dataset-sts
 - cloned locally
 - para dataset seems good

https://www.kaggle.com/rishisankineni/text-similarity

http://ixa2.si.ehu.es/stswiki/index.php/Main_Page

http://alt.qcri.org/semeval2014/task3/index.php?id=data-and-tools



# Other 

https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/

https://keras.io/getting-started/functional-api-guide/#shared-layers

https://gist.github.com/mmmikael/0a3d4fae965bdbec1f9d

https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.2-understanding-recurrent-neural-networks.ipynb

