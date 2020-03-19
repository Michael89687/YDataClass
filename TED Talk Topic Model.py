#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datascience import *
import numpy as np
import re
import gensim

from collections import Counter

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logging.root.level = logging.CRITICAL 

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# direct plots to appear within the cell, and set their style
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')


# In[2]:


filename = "https://s3.amazonaws.com/sds171/labs/lab07/ted_talks.csv"
data = Table.read_table(filename)

transcripts = data.column('transcript')


# In[3]:


#Using regular expression to clean the data
transcripts = [re.sub('-', ' ', plot) for plot in transcripts]
transcripts = [re.sub('[^\w\s]', '', plot) for plot in transcripts]
transcripts = [re.sub('[A-Z]\w*', '', plot) for plot in transcripts]
transcripts = [re.sub('[ ]+', ' ', plot) for plot in transcripts]


# In[4]:


def is_numeric(string):
    for char in string:
        if char.isdigit():
            return True
    return False

def has_poss_contr(string):
    for i in range(len(string) - 1):
        if string[i] == '\'' and string[i+1] == 's':
            return True
    return False

def empty_string(string):
    return string == ''

def remove_string(string):
    return is_numeric(string) | has_poss_contr(string) | empty_string(string)


# In[5]:


#Tokenize
plots_tok = []
for plot in transcripts:
    processed = plot.lower().strip().split(' ')
    plots_tok.append(processed)

#Removing numeric, posessives/contractions, and empty strings
temp = []
for plot in plots_tok:
    filtered = []
    for token in plot:
        if not remove_string(token):
            filtered.append(token)
    temp.append(filtered)
plots_tok = temp


# In[6]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#Lemmatizing the tokens 
lemmatizer = WordNetLemmatizer()

temp = []
for plot in plots_tok:
    processed = []
    for token in plot:
        processed.append(lemmatizer.lemmatize(token, pos='v'))
    temp.append(processed)
plots_tok = temp


# In[7]:


#Creating the Counter
vocab = Counter()
for plot in plots_tok:
    vocab.update(plot)

print("Number of unique tokens: %d" % len(vocab))


# In[8]:


#Keeping tokens that appear more than 20 times 
tokens = []
for token in vocab.elements():
    if vocab[token] > 20:
        tokens.append(token)
vocab = Counter(tokens)

print("Number of unique tokens: %d" % len(vocab))


# In[9]:


#Removing rare and stop words
stop_words = []
for item in vocab.most_common(200):
    stop_word = item[0]
    stop_words.append(stop_word)
tokens = []
for token in vocab.elements():
    if token not in stop_words:
        tokens.append(token)
vocab = Counter(tokens)

print("Number of unique tokens: %d" % len(vocab))


# In[10]:


#Creating the identifier mappings word2id and id2word
items = vocab.items()
id2word = {}
word2id = {}
idx = 0
for word, count in vocab.items():
    id2word[idx] = word
    word2id[word] = idx
    idx += 1
    
print("Number of tokens mapped: %d" % len(id2word))
print("Identifier for 'photograph': %d" % word2id['photograph'])
print("Word for identifier %d: %s" % (word2id['photograph'], id2word[word2id['photograph']]))


# In[11]:


#Filtering the tokens 
temp = []
for plot in plots_tok:
    filtered = []
    for token in plot:
        if token in vocab:
            filtered.append(token)
    temp.append(filtered)
plots_tok = temp


# In[12]:


#Creating the Corpus
sample = 30
corpus = []
for plot in plots_tok:
    plot_count = Counter(plot)
    corpus_doc = []
    for item in plot_count.items():
        pair = (word2id[item[0]], item[1])
        corpus_doc.append(pair)
    corpus.append(corpus_doc)

print("Plot, tokenized:\n", plots_tok[sample], "\n")
print("Plot, in corpus format:\n", corpus[sample])


# In[13]:


get_ipython().run_cell_magic('time', '', "lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,\n                                            id2word=id2word,\n                                            num_topics=10, \n                                            random_state=100,\n                                            update_every=1,\n                                            chunksize=100,\n                                            passes=10,\n                                            alpha='auto',\n                                            per_word_topics=True)")


# In[14]:


num_topics = 10
num_words = 15
top_words = Table().with_column('word rank', np.arange(1,num_words+1))
for k in np.arange(num_topics): 
    topic = lda_model.get_topic_terms(k, num_words)
    words = [id2word[topic[i][0]] for i in np.arange(num_words)]
    probs = [topic[i][1] for i in np.arange(num_words)]
    top_words = top_words.with_column('topic %d' % k, words)
    
top_words.show()


# In[15]:


sample = 13
topic_dist = lda_model.get_document_topics(corpus[sample], minimum_probability = 0)
topics = [pair[0] for pair in topic_dist] 
probabilities = [pair[1] for pair in topic_dist]
topic_dist_table = Table().with_columns('Topic', topics, 'Probabilities', probabilities)
topic_dist_table.show(20)
t = np.argmax(probabilities)
print("Topic with highest probability: %d (%f)" % (t, probabilities[t]))


# In[16]:


print(transcripts[sample][0:2500])


# In this example, Topic 7, which represents technology, has the highest probability with .646. Looking at the transcript of the talk, we see that this is in fact true. In the transcript of this sample, we see terms like "screensaver", "touch sensor", and "interactive".

# In[17]:


sample = 7
topic_dist = lda_model.get_document_topics(corpus[sample], minimum_probability = 0)
probabilities = [pair[1] for pair in topic_dist]
topics = [pair[0] for pair in topic_dist]
topic_dist_table = Table().with_columns('Topic', topics, 'Probabilities', probabilities)
topic_dist_table.show(20)
t = np.argmax(probabilities)
print("Topic with highest probability: %d (%f)" % (t, probabilities[t]))


# In[18]:


print(transcripts[sample][0:2500])


# In this sample, we observe that topic 8, which represents art, has the highest probability with .51. Looking at the transcript, we can see that our topic model is correct since there are terms like "modernists", "design", and "diagram".

# In[19]:


sample = 31
topic_dist = lda_model.get_document_topics(corpus[sample], minimum_probability = 0)
probabilities = [pair[1] for pair in topic_dist]
topics = [pair[0] for pair in topic_dist]
topic_dist_table = Table().with_columns('Topic', topics, 'Probabilities', probabilities)
topic_dist_table.show(20)
t = np.argmax(probabilities)
print("Topic with highest probability: %d (%f)" % (t, probabilities[t]))


# In[20]:


print(transcripts[sample][0:2500])


# The topic with the highest probability of .64 is topic 4, which represents medicine. Looking at the transcript of the talk, we observe that our topic model was able to correctly identify the topic at hand. Terms like "cancer", "medical", and "research" were all used in this TED talk.
