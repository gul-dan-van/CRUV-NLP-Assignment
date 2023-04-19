import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import textstat
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import math
import re
from stop_words import get_stop_words
from sent2vec.vectorizer import Vectorizer
from transformers import BertTokenizer, BertModel

nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(get_stop_words('en'))

df = pd.read_csv('news.csv')

df = df.sample(len(df)) # Shuffling the data
train = df.iloc[:int(len(df)*0.9),:]
test = df.iloc[int(len(df)*0.9):,:]

content=df.content[0]
title=df.title[0]

# Preprocessing function used for cleaning passages for creating embedding vectors

def preprocessing(para, stemmize=False, keep_sw=False):
  w_token = word_tokenize
  s_token = sent_tokenize
  lem = WordNetLemmatizer()
  stemmer = PorterStemmer()
  lst = []
  if type(para) == str:
    lst = [x[:-1] for x in sent_tokenize(para)]
  else:
    lst = [x for x in para]
  lst = [word_tokenize(re.sub('[^a-z .]','',sent)) for sent in lst]

  if not keep_sw:
    lst = [
      [x for x in sent if x.lower() not in stop_words]
      for sent in lst
    ]

    if stemmize:
      lst = [
              [stemmer.stem(word) for word in sent]
              for sent in lst
          ]
    else:
      lst = [
          [lem.lemmatize(word) for word in sent]
          for sent in lst
      ]
  lst = [' '.join(sent) for sent in lst]

  lst = [word_tokenize(x) for x in lst]
  lst = [[x, ['']][len(x)==0] for x in lst]

  return lst

# BM25 Model to compare TF-IDF score of content with its title to tell relevant sentences

class BM25:
  def __init__(self, k1=1.2, b=0.75):
    self.b = b
    self.k1 = k1

  def fit(self, corpus):
    tf = []
    df = {}
    idf = {}
    doc_len = []
    corpus_size = 0
    for document in corpus:
      corpus_size += 1
      doc_len.append(len(document))

      # compute tf (term frequency) per document
      frequencies = {}
      for term in document:
        term_count = frequencies.get(term, 0) + 1
        frequencies[term] = term_count

      tf.append(frequencies)

      # compute df (document frequency) per term
      for term, _ in frequencies.items():
        df_count = df.get(term, 0) + 1
        df[term] = df_count

    for term, freq in df.items():
      idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

    self.tf_ = tf
    self.df_ = df
    self.idf_ = idf
    self.doc_len_ = doc_len
    self.corpus_ = corpus
    self.corpus_size_ = corpus_size
    self.avg_doc_len_ = sum(doc_len) / corpus_size
    return self

  def search(self, query):
    scores = [self._score(query, index) for index in range(self.corpus_size_)]
    return scores

  def _score(self, query, index):
    score = 0.0

    doc_len = self.doc_len_[index]
    frequencies = self.tf_[index]
    for term in query:
      if term not in frequencies:
        continue

      freq = frequencies[term]
      numerator = self.idf_[term] * freq * (self.k1 + 1)
      denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
      score += (numerator / denominator)

    return score

# Defining a class to encapsulate all defined sentence filtering functions and their performance.
# This will make it very simple to use and compare different defined functions and to add many more in future

class CleanPassage():
  def __init__(self, passage, title):
    self.title = title
    self.passage = passage
    self.filtered_passage = passage
    self.sentences = nltk.sent_tokenize(passage)
    self.filters = {}

    self.available_metrics = [
        'reduced_length',
        'flesch_kincaid_grade_level',
        'gunning_fog_index',
        'cosine_similarity',
        # 'topic_coherence',
        'named_entities_diff'
    ]

    self.embedding_functions = [
        'sent2vec'
    ]

  def list_functions(self):
    '''Use this function to print all availabel functions used for filtering sentences'''

    # print(*self.filters.keys(), sep='\n')
    return [*self.filters.keys()]
  
  def list_metrics(self):
    '''Use this function to list all metrics being used'''

    # print(*self.available_metrics, sep='\n')
    return [*self.available_metrics]

#################################### FILTER FUNCTIONS ####################################

  '''
      These functions will have a general strategy of calculating a score for each sentence and remove those sentence who has score too small as compared to the mean, or has large mahalanobis distance
  '''


  def tfidf_filter(self, thres=None, use_norm=False):
    '''
        Summing over TF-IDF score of words in a sentence to calculate  TF-IDF score of the sentence.
        Then removing sentences with score below a threshold using mahalanobis distancce
    '''

    filter_name = 'tfidf_filter'
    self.filters[filter_name] = {}

    vectorizer = TfidfVectorizer()

    sentences = nltk.sent_tokenize(self.passage)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_array = tfidf_matrix.toarray()
    sentence_scores = tfidf_array.sum(axis=1)

    if use_norm:
      if thres==None:
        thres = 0.2
      norm_sentence_scores = sentence_scores / np.linalg.norm(sentence_scores)
      filtered_sentences = [sentences[i] for i in range(len(sentences)) if norm_sentence_scores[i] > thres]
      filtered_out_sentences = [sentences[i] for i in range(len(sentences)) if norm_sentence_scores[i] <= thres]

    else:
      if thres == None:
        thres = 2
      mean_score = sentence_scores.mean()
      score_variance = sentence_scores.var()
      delta = np.abs(sentence_scores - mean_score)
      mahalanobis_distances = np.sqrt(np.square(delta)/score_variance)

      self.filters[filter_name]['selected sentences'] = [sentences[i] for i in range(len(sentences)) if mahalanobis_distances[i] < thres]
      self.filters[filter_name]['rejected sentences'] = [sentences[i] for i in range(len(sentences)) if mahalanobis_distances[i] >= thres]

    self.filters[filter_name]['passage'] = self.passage
    self.filters[filter_name]['selected passage'] = ' '.join(self.filters[filter_name]['selected sentences'])


  def bm25_filter(self, thres=2):
    '''
        Using BM25 to calculate scores for the sentences
    '''

    filter_name = 'bm25_filter'
    self.filters[filter_name] = {}

    sentences = preprocessing(self.passage)
    query = preprocessing(self.title)[0]

    model = BM25()
    model.fit(sentences)

    scores = model.search(query)
    best_sents = []
    for i in range(len(scores)):
      best_sents.append([scores[i], i])

    best_sents.sort()
    sentence_scores = np.array([
        x[0] for x in best_sents
    ])

    mean_score = sentence_scores.mean()
    score_variance = sentence_scores.var()
    delta = np.abs(sentence_scores - mean_score)
    mahalanobis_distances = np.sqrt(np.square(delta)/score_variance)

    self.filters[filter_name]['selected sentences'] = [self.sentences[best_sents[i][1]] for i in range(len(self.sentences)) if mahalanobis_distances[i] < thres]
    self.filters[filter_name]['rejected sentences'] = [self.sentences[best_sents[i][1]] for i in range(len(self.sentences)) if mahalanobis_distances[i] >= thres]
    self.filters[filter_name]['passage'] = self.passage
    self.filters[filter_name]['selected passage'] = ' '.join(self.filters[filter_name]['rejected sentences'])


  def combined_filter(self, thres=2):
    '''
        using different scoreing methods to calculate scores for a sentence
        using those different scores to calculate total weighted score for a sentence
    '''

    filter_name = 'combined_filter'
    self.filters[filter_name] = {}

    nlp = spacy.load('en_core_web_sm')

    sent_ents = [len(nlp(sent).ents) for sent in self.sentences] # number of named entities in each sentence
    total_ents = sum(sent_ents)+1
    scores = []

    # Calculating scores for each sentence using different methods, and then giving weights (importance) to different scores
    n = len(self.sentences)
    for i in range(n):
      total_score = 0.5 * sent_ents[i]/total_ents + 0.5 * sent_ents[i]/(len(self.sentences[i].split())+1)
      scores += [total_score]

    mean_score = np.mean(scores)
    score_variance = np.var(scores)
    delta = np.abs(np.array(scores) - mean_score)
    mahalanobis_distances = np.sqrt(np.square(delta)/score_variance)

    self.filters[filter_name]['selected sentences'] = [self.sentences[i] for i in range(len(self.sentences)) if mahalanobis_distances[i] < thres]
    self.filters[filter_name]['rejected sentences'] = [self.sentences[i] for i in range(len(self.sentences)) if mahalanobis_distances[i] >= thres]

    self.filters[filter_name]['passage'] = self.passage
    self.filters[filter_name]['selected passage'] = ' '.join(self.filters[filter_name]['selected sentences'])


#################################### EMBEDDING FUNCTIONS ####################################


  def sent2vec_embed(self):
    
    vectorizer = Vectorizer(pretrained_weights='distilbert-base-multilingual-cased')
    vectorizer.run(self.sentences, remove_stop_words=['not'], add_stop_words=[])
    vectors = vectorizer.vectors

    return self.sentences


#################################### METRIC FUNCTIONS ####################################


  def get_reduced_length(self, filter):
    ''' 
        It is possible that a sentence is being restructured and then being repeated for the entire paragraph. 
        So, in some cases, ratio of final_length over initial_length tending to zero is required result.
    '''

    sentences = nltk.sent_tokenize(self.filters[filter]['passage'])
    filtered_sentences = nltk.sent_tokenize(self.filters[filter]['selected passage'])
    return len(filtered_sentences)/len(sentences) # Smaller value is preferred

  def get_flesch_kincaid_grade_level(self, filter):
    ''' 
        In short FKGL tells about the readability of the text.
        Its formula states that more syllables per word and more words per sentence is better.
        But too many syllables per word or too many words per sentence might degrade the quality.
        So, good FKGL score is 7.2 according to the American standards.

        FKGL = 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
    '''

    before = abs(textstat.flesch_kincaid_grade(self.filters[filter]['passage']) - 7.2)
    after = abs(textstat.flesch_kincaid_grade(self.filters[filter]['selected passage']) - 7.2)
    return abs(after-before)/before # Smaller value is preferred

  def get_gunning_fog_index(self, filter):
      '''
          GFI is similar to FKGL conceptually, so, both give similar knowledge about the paragraph, but give different weights to the variables
          GFI = 0.4 * ((total words / total sentences) + (percentage of words with 3 or more syllables))
      '''

      before = textstat.gunning_fog(self.filters[filter]['passage'])
      after = textstat.gunning_fog(self.filters[filter]['selected passage'])
      return abs(after-before)/before # Smaller value is preferred

  def get_cosine_similarity(self, filter):
      '''
          Checking cosine *distance* between passage before and after the update
      '''

      vectorizer = TfidfVectorizer()
      tfidf_matrix = vectorizer.fit_transform([self.filters[filter]['passage'], self.filters[filter]['selected passage']])
      similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
      return 1 - similarity[0][0] # Smaller value is preferred

  def get_topic_coherence(self, filter, num_topics=5):
      '''
         Using Latent Dirichlet Allocation (LDA),
         which is a probabilistic model to get topic distribution given a document, and word distribution given a topic,
         I am giving the perplexity scores to the sentences based on the likelihood of those sentences being generated by the LDA model when the topic is provided.
      '''

      vectorizer = TfidfVectorizer()
      tfidf_matrix = vectorizer.fit_transform([self.filters[filter]['passage']])

      lda = LatentDirichletAllocation(n_components=num_topics)
      lda.fit(tfidf_matrix)
      before = lda.score(tfidf_matrix)
      
      tfidf_matrix = vectorizer.fit_transform([self.filters[filter]['selected passage']])

      lda = LatentDirichletAllocation(n_components=num_topics)
      lda.fit(tfidf_matrix)
      after = lda.score(tfidf_matrix)
      return abs(before-after)/before # Smaller values are preferred

  def get_named_entities_diff(self, filter):
      '''
          Counting change in number of named entities mentioned.
          This change should be small.
      '''

      nlp = spacy.load('en_core_web_sm')
      
      doc = nlp(self.filters[filter]['passage'])
      passage_named_entities = []
      for entity in doc.ents:
          passage_named_entities.append(entity.text)

      before = len(passage_named_entities)
      
      doc = nlp(self.filters[filter]['selected passage'])
      selected_passage_named_entities = []
      for entity in doc.ents:
          selected_passage_named_entities.append(entity.text)

      after = len(selected_passage_named_entities)
      delta = before - after
      return delta / len(passage_named_entities) # Smaller values are preferred

  
  def calculate_metrics(self):
    metrics = {}
    for filter in self.filters:
      metrics[filter] = {}
      metrics[filter]['reduced_length'] = self.get_reduced_length(filter)
      metrics[filter]['flesch_kincaid_grade_level'] = self.get_flesch_kincaid_grade_level(filter)
      metrics[filter]['gunning_fog_index'] = self.get_gunning_fog_index(filter)
      metrics[filter]['cosine_similarity'] = self.get_cosine_similarity(filter)
      # metrics[filter]['topic_coherence'] = self.get_topic_coherence(filter)
      metrics[filter]['named_entities_diff'] = self.get_named_entities_diff(filter)

    return metrics

c = CleanPassage(content, title)
c.combined_filter()
c.calculate_metrics()

columns = ['Original Content', 'New Content', 'Removed Lines']
columns += c.list_metrics()

resulting_data = {
    col: [] for col in columns
}

filter = 'combined_filter'
for i, row in test.iterrows():
  oc = row['content'] # Original Content
  title = row['title']

  c=CleanPassage(oc, title)
  c.combined_filter()
  scores = c.calculate_metrics()[filter]
  for metric in scores.keys():
    resulting_data[metric] += [scores[metric]]

  nc = c.filters[filter]['selected passage'] # New Content
  rl = c.filters[filter]['rejected sentences'] # Removed lines

  resulting_data['Original Content'] += [oc]
  resulting_data['New Content'] += [nc]
  resulting_data['Removed Lines'] += [rl]

result_df = pd.DataFrame(data = resulting_data)
result_df.to_csv('results.csv')