# LAB 1 - Feature-Based Classification
## Sentiment Analysis
### Loading the dataset
#### Exercices:
##### 1. Why does HuggingFace download the corpora into a cache? 

HuggingFace downloads the corpora into a cache to ensure it is not downloaded again if it was already done before and that it was not updated. In case of an update, it will keep the previous files but will download only the new one, according to HuggingFace documentation (https://huggingface.co/docs/huggingface_hub/guides/manage-cache).


##### 2. Plot the distribution of labels of test and the train set. Interpret.

In the train set, the distribution of labels is exactly equivalent between label 0 and 1. For each, there are 180 000 labels, that fits the total number of 3 600 000 examples. There is the same distribution for the test set, with 200 000 labels of each, totalling 400 000 annotated examples. The distribution is equivalent for both.
 

##### 3. What are the 5 most common words in the title? (We will use a simple tokenization using spaces to identify word and lowercasing as the sole preprocessing). Interpret. 
#####

from datasets import load_dataset
dataset = load_dataset("amazon_polarity")

##### 4. Why is the corpus split into a train and a test set (rather than letting users splitting the data)? Is this a good idea?
