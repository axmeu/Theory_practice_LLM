# LAB 1 - Feature-Based Classification
## Sentiment Analysis
### Loading the dataset
#### 1. Why does HuggingFace download the corpora into a cache? 

HuggingFace downloads the corpora into a cache to ensure it is not downloaded again if it was already done before and that it was not updated. In case of an update, it will keep the previous files but will download only the new one, according to HuggingFace documentation (https://huggingface.co/docs/huggingface_hub/guides/manage-cache).


#### 2. Plot the distribution of labels of test and the train set. Interpret.
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
 

#### 3. What are the 5 most common words in the title? (We will use a simple tokenization using spaces to identify word and lowercasing as the sole preprocessing). Interpret. 

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


#### 4. Why is the corpus split into a train and a test set (rather than letting users splitting the data)? Is this a good idea?

It was split into a train and test set to ensure different trained models can be evaluated on the same data, plus there are equally distributed labels for both set. This is a good idea so different models can be compared more rigourously, being trained on the same data or at least evaluated on the same test set. 
However, by having an equally distributed labels, it could be a bad idea in the case the real data this models has to be used on doesn't include this kind of distribution; for example when the models has to look for spam vs non-spam emails, we can imagine emails are more likely non-spam. 

### Extracting features
#### 5. Using `sklearn`'s `CountVectorizer` build the observation matrix corresponding the train & the test sets

``` python


```


``` text
# output

```


#### 6. Do you need to tokenize the data first (e.g. by separating the punctuation first)?



#### 7. What are stop words and why we should not consider them as features?



#### 8. Do we need to use a pre-computed list of stop words? (look at the `max_df` parameter)



#### 9. Why should we set the `min_df` value to 2?



#### 10. Compute the vocabulary size when `min_df` is set to 1 or to 2.

``` python


```


``` text
# output

```