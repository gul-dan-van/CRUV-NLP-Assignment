# CRUV NLP Assignment

**By Gauranshu**

<br><br>

I was provided with a dataframe containing passages and their details. I was assigned a task to remove some lines from the passages that are not too significant, and therefore, shortening the passage. Task contained different aspects of NLP, text ranking, sentence embedding, semantic analysis and many more.
<br>
I started by using a BM25 model I used once for my Inter IIT project for information retrieval. I also used a plain tfidf filter which works similar to BM25. After getting scores for each sentence, I implemented two different thresholding methods to filter out less scoring sentences. First was very simple, I just removed sentences with scores less than a specific value, this threshold value is decided manually by analysing the results. Second method I used was Mahalanobis Distance, which is a distance metric of a data point, from the mean point in using standard deviations as units of distance. Roughly, Mahalanobis distance could be considered as percentile metrics, as Mahalanobis distance less than 3 means top ~97% data points closest to the mean. I rejected sentences with a mahalanobis distance greater than 2. I planned a third method, a filter that combines different methods. Currently, all I have added is just ranking on the basis of importance of named entities contained in the sentence.
<br>
For metrics, I used
- **reduced length metric** to give the ratio shrinkage of the resulting passage
- **Flesch kincaid grade level and Gunnig fog index** to calculate readability of the passage
- **cosine similarity between tfidf vectors**
- **named entities difference** to know the ratio change in the number of named entities.

#### More to come..
I have also added feature of sentence embeddings. I can use these embedding to train a clustering algorithm and remove sentences which lie far from clusters. I focused more on TF-IDF score, what I can do is use more of semantic features for feature engineering and train some weights to give more importance to different features.
