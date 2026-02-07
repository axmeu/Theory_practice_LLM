# **LAB 1 - Feature-Based Classification**
## **Sentiment Analysis**
### **Loading the dataset**
#### **1. Why does HuggingFace download the corpora into a cache?**

HuggingFace downloads the corpora into a cache to ensure it is not downloaded again if it was already done before and that it was not updated. In case of an update, it will keep the previous files but will download only the new one, according to HuggingFace documentation (https://huggingface.co/docs/huggingface_hub/guides/manage-cache).


#### **2. Plot the distribution of labels of test and the train set. Interpret.**
``` python
df_train = pd.DataFrame(dataset["train"])
df_train.groupby("label").size().plot(kind="bar")
```
![Alt texte](train_label.png)

``` python
df_test = pd.DataFrame(dataset["test"])
df_test.groupby("label").size().plot(kind="bar")
```
![Alt texte](test_label.png)

In the train set, the distribution of labels is exactly equivalent between label 0 and 1. For each, there are 180 000 labels, that fits the total number of 3 600 000 examples. There is the same distribution for the test set, with 200 000 labels of each, totalling 400 000 annotated examples. The distribution is equivalent for both.
 

#### **3. What are the 5 most common words in the title? (We will use a simple tokenization using spaces to identify word and lowercasing as the sole preprocessing). Interpret.**

``` python
from collections import Counter
print(
    Counter(
        " ".join(
            df_train["title"]
                .str.lower()
                .values
        ).split()
    ).most_common(5)
)
```
``` text
# output
[('the', 488026), ('a', 423730), ('not', 298203), ('of', 257353), ('great', 251930)]
```

Ths 5 most common words in the title (considering only train set) are the words "the", "a", "not", "of" and "great". They are expected words as they are english stopwords (very frequent words), as no removal has been done on the text. 


#### 4. **Why is the corpus split into a train and a test set (rather than letting users splitting the data)? Is this a good idea?**

It was split into a train and test set to ensure different trained models can be evaluated on the same data, plus there are equally distributed labels for both set. This is a good idea so different models can be compared more rigourously, being trained on the same data or at least evaluated on the same test set. 
However, by having an equally distributed labels, it could be a bad idea in the case the real data this models has to be used on doesn't include this kind of distribution; for example when the models has to look for spam vs non-spam emails, we can imagine emails are more likely non-spam. 

### **Extracting features**
#### **5. Using `sklearn`'s `CountVectorizer` build the observation matrix corresponding the train & the test sets**

``` python
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df_train["content"])
X_test = vectorizer.transform(df_test["content"])
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_train.shape}")
```
``` text
# output
X_train shape: (3600000, 937124)
X_test shape: (3600000, 937124)
```
Only the shape was printed as it would cause a memory problem to display whole matrices with a `.toarray()`call. 

#### **6. Do you need to tokenize the data first (e.g. by separating the punctuation first)?**

It is not necessary to tokenize the data first as sklearn can do it, but it leaves us the option to apply our own tokenizer. 


#### **7. What are stop words and why we should not consider them as features?**

Stop words are words that are over represented and not carrying a lot of information (such as short grammatical words). They should not be considered as features as they will extand the number of dimension of our matrices while not being really usefull for the task we want to do. 

#### **8. Do we need to use a pre-computed list of stop words? (look at the `max_df` parameter)**

It is not necessary as in sklearn library they can be handled by setting a `max_df` parameter that can be either between 0 and 1 to set a percentage of maximum document frequency for the words we want to eliminate, or  above 1 to set a number of most represented words to eliminate.

#### **9. Why should we set the `min_df` value to 2?**

We should do it so any word that is occuring in less than 2 document are eliminated. It is usefull as they're most likely hapax and might extenda lot the vocabulary (for example if we have even a typo in a document it would count as a word of our vocabulary).

#### **10. Compute the vocabulary size when `min_df` is set to 1 or to 2.**

By running the code given in question 5, adding the code below,

``` python
print(len(vectorizer.vocabulary_))
```
we obtain: 

`min_df=1` : vocabulary size of 937124.

`min_df=2` : vocabulary size of 355610.


### **Training & Evaluating a Classifier**
#### **11. Using the data you have just prepared, train a logistic regression considering as features either the content or the title Evaluate the accuracy of your classifier on the train & on the test set and plot the confusion matrix. Interpret.**



#### **12. How can you use your classifier to estimate the confidence in the predicted label? Plot the confidence distribution for example that are correctly identified and examples that are misclassified.**

#### **13. Using l1 regularization, identify the 100 most relevant words identified by the model. You should consider the l1_min_c function**

#### **14. Is it reasonable to set min_df at 2? Why?**
