# Papers
Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks: http://aclweb.org/anthology/D/D15/D15-1181.pdf
    
    -Implementation in torch https://hohocode.github.io/textSimilarityConvNet/


Learning discriminative projections for text similarity measures https://dl.acm.org/citation.cfm?id=2018965 

    - Talks about how TF-IDF measure works well but doesn't handle words that have similar semantic meaning well
    - Learns a projection of the word vector to a low dimmensonal space so the semantic relationships still hold
    - Simple setup, first layer is just the word vec input, 2nd layer is out put. Uses no activation function so it is just a linear projection
    - Find similairty with cosine distance btwn the projected vectors
    - Loss is pairwise loss instead of looking at individual pairs, this makes it better at ranking instead of computing direct similairty 

# Models

https://github.com/dhwajraj/deep-siamese-text-similarity


# Data
most do similar not similar instead of real number similairty 
https://nlp.stanford.edu/projects/snli/

https://github.com/brmson/dataset-sts
 - cloned locally
 - para dataset seems good