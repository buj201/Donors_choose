import pandas as pd
import os
import pyLDAvis
import pyLDAvis.sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Get data
os.system('curl http://s3.amazonaws.com/open_data/opendata_essays000.gz -o opendata_essays000.gz')

#Select essays from rows with no missing values
essays = pd.read_csv('opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])

essays = essays.dropna(axis=0, how='any').essay

#Featurize essays- for initial exploration just use term frequency
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b',
                                max_df = 0.5,
                                min_df = 10)
dtm_tf = tf_vectorizer.fit_transform(essays)

#Now fit LDA model- note this is unsupervised LDA- not full model yet.
# For initial exploration we'll use 6 categories based on number of resource_type categories:
# Books, Technology, Supplies, Trips, Visitors, Other
# See https://research.donorschoose.org/t/opedata-layout-and-docs/18

lda_tf = LatentDirichletAllocation(n_topics=6, random_state=0)
lda_tf.fit(dtm_tf)
pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)

#Save visualization for submission
PreparedData = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
pyLDAvis.save_html(PreparedData, 'DonorsChoose.html')

