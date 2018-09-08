# 421-honors

Links to papers with notes are in the resources folder


# Wrap up Paper

## Goal

The goal of this project was to compare how well different methods of sentence similarity work on different datasets. The two methods I tested were TF-IDF score and using a Siamese Neural Network. I hypothesized that while the TF-IDF score would be easy to compute, it would not function well on actual data due to its simplicity were as the neural network would take a while to train but would ultimately have better accuracy. 

## Data 

There were two datasets used to test the different methods. The first was The Stanford Natural Language Inference (SNLI) Corpus. SNLI had 367,373 sentence pairs with a Boolean score for if the pair of sentences was similar or not. The second dataset was the Semantic Textual Similarity (STS) benchmark. This dataset consisted of about 5,749 sentence pairs for training, 1,500 for validation, and 1,379 for testing. Each with a 0-4 score for how similar they were. These datasets consist of sentences from multiple domains, so they offer a good baseline for how well the model works. 

## TF-IDF

TF-IDF stands for term frequency inverse document frequency. This works by going through all the sentences in the dataset and count the number of times each word is seen. Then you turn each sentence into a word frequency vector. You then weight each element of the vector by how often that word is seen in the whole dataset. So, if a word appears in many sentences it would have a lower value in each word frequency vector it appears in were as if it is seen in only one sentence it would have a weight of 1. After computing these vectors, you finally do cosine similarity to see how similar the sentences are. This method’s main issue is that is relies only on the frequency of the words, not the context of the words. It also loses any possible similarity of words. For example, car and vehicle are very similar words but are treated completely differently by TF-IDF. 

## Siamese Neural Net

 A Siamese net is a neural network that has 2 inputs, the layers attached to the inputs have the same weights and are then merged together before outputting. The input layers are structured as follows, an embedding layer which is followed by an LSTM layer. The embedding layer will learn any of the similarities of words so things like car and vehicle will have similar vectors. The LSTM (Long Short-Term Memory) is a recurrent layer that is designed to hold some of the information of the previous words it sees. This will help the network understand the context each word is used in. After the LSTM layer, the network will output a vector that represents the sentence with context, and word similarity taken into account. Finally, these sentence representations are merged together with two different methods, cosine similarity, and just concatenating the 2 vectors. Then finally, sigmoid activation is added to output a probability of how similar the sentences are. When training the network, I used early stopping so the net would stop training once the accuracy of the validation set got worse. 

## Results

The TF-IDF did not fare well. I only tested it on a subset of the SNLI dataset and it had an accuracy of 0.01, since this score was so bad I did not try to test it any further to get any improvements.  

| SNLI Neural Net Accuracy | GLOVE  | Learned Embeddings |
|--------------------------|--------|--------------------|
| Cosine merge             | 72.66% | 73.27%             |
| Concatenate              | 78.10% | 77.86%             |

| STS Neural Net Mean Absolute Error (Only using GLOVE) |      |
|-------------------------------------------------------|------|
| Cosine Merge                                          | 0.96 |
| Concatenate                                           | 1.33 |

## Conclusion

The TF-IDF approach was about as bad as an approach can be. Due to its simplicity it could not handle the subtly of the similarities of the sentence pairs the datasets provide. The NNs however faired quite well. These were quite simple networks that did not require much code (thanks to the ease of use of Keras). The training was fast (about 20min for SNLI, about 30 secs for STS on a 2015 MacBook). I was surprised that the cosine merge did not fare as well as just concatenating the two embedded vectors together on the SNLI dataset. I assume that since the output of the concatenate is a 200-length vector, the network had more weights it could tune on the output node compared to just 1 with the cosine merge. However, the STS dataset did better with Cosine merge, so more investigation would have to be done to see the reason for this. With more tweaking I could easily see the accuracy getting much better with the NN models.

## References

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. 

Neculoiu, P., Versteegh, M., & Rotaru, M. (2016). Learning Text Similarity with Siamese Recurrent Networks. Proceedings of the 1st Workshop on Representation Learning for NLP. doi:10.18653/v1/w16-1617

STSbenchmark. (n.d.). Retrieved from http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark

Tf-Idf and Cosine similarity. (2016, November 16). Retrieved from https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
