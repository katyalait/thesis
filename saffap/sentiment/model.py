import gensim
import re
import pandas as pd
from time import time
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy.lang.en import English
import logging
from gensim.models.phrases import Phrases, Phraser
import sys
from data_handler.models import StockPrice, Article, Asset, Source
from sentiment.models import Category, Label, Column, Value
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.interfaces import TransformedCorpus
import os
from data_handler.helpers import progress, time_to_complete, daterange
from datetime import timedelta, date, datetime
from plotly.offline import plot
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_handler.iterators import SentimentIterator, SentimentExtractor
from statsmodels.tsa.stattools import adfuller
import hurst as hrst
from scipy import stats
from django.db.models import Avg, Count, Min, Sum, Max



logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA = os.path.join(BASE, "cleaned_data.csv")
MODEL = os.path.join(BASE, "model2vec.model")
CORPUS = os.path.join(BASE, "corpus.mm")

class DocModelGenerator(object):
    """
    DocModelGenerator is an object which holds the predefined methods for
    creating a doc2vec model for a corpus of articles. By aggregating the
    necessary preprocessing and data cleaning methods, it reduces the overall
    complexity of desiging a doc2vec model when compared to other manual
    methods.

    Creates a doc2vec model for a given input csv file which contains headline,
    date and contents columns. Names for these columns can also be provided in
    the init but are set to 'headline', 'date' and 'content' by default.
    param file_path: a path to the .csv file containing the data
    param headline: name for the column which contains headline data
    param data: name for the column which contains date data
    param contents: name for the column which contains contents data
    """

    class Rating:
        def __init__(self, article, rating):
            self.article = article
            self.rating = rating

    class _doc_iter:
        def __init__(self, labels_list, doc_labels):
           self.doc_labels = labels_list
           self.doc_list = clean_docs
        def __iter__(self):
            for index, row in self.doc_list.iteritems():
                yield TaggedDocument(words = row.split(), tags=[self.labels_list[index]])

    class _article_iter:
        def __init__(self, similar):
            self.similar = similar

        def __iter__(self):
            for headline, rating in similar:
                article = Article.objects.get(headline=headline)
                yield Rating(article, rating)



    def __init__(self):
        logging.info("Ready. Create or load a model.")

    def load_model(self, path):
        full_path  = os.path.join(BASE_DIR, path)
        self.model = Doc2Vec.laod(full_path)


    def load_csv(self, file_path, headline='headline',
                    date='date', content='content'):
        self.df = pd.read_csv(filepath, usecols=[headline, date, content]).dropna()
        self.df = self.df.dropna().reset_index(drop=True).drop_duplicates()
        logging.info("Generator ready. Call create_model() to continue ...")

    def token_clean(doc):
        text = [token.lemma_ for token in doc if not token.is_stop]
        if len(text)>2:
            text = ' '.join(text)
            return text

    def preprocess(self, to_process):
        nlp = spacy.load("en_core_web_sm")
        logging.info("Prepaing data ...")
        t = time()
        index = 0
        mini_clean = (re.sub("[^A-Za-z']+", ' ', str(entry)).lower()
                        for entry in to_process)
        self.clean_docs = [token_clean(doc) for doc in
                            nlp.pipe(mini_clean,
                            disable=["tagger", "parser", "ner"])]
        print("Time to clean up everything: {}".format((time() - t)/ 60))

    def create_iterator(self):
        self.doc_labels = self.df['headline']
        self.preprocess((self.df['headline'] + ". " + self.df['content']))
        self.dociter = _doc_iter(self.doc_labels, self.clean_docs)
        return self.dociter

    def create_model(self, **kwargs):
        self.model = Doc2Vec(**kwargs)
        return self.model

    def build_vocab(self, **kwargs):
        self.model.build_vocab(self.dociter, **kwargs)
        return self.model

    def train_model(self, **kwargs):
        self.model.train(**kwargs)
        self.model.save(os.path.join(BASE_DIR, "models/doc2vec.model"))
        return self.model

    def get_related_articles(self, article, topn=20, ratings=True):
        similar = self.model.docvecs.most_similar(article.headline, topn=topn)
        return self._article_iter(similar)






class SentimentPriceModel(object):
    sentiments = Category.objects.all().values_list('name', flat=True)

    class StationaryResults():
        def __init__(self, df, column, adf, hurst):
            self.df = df[column].dropna()
            if adf:
                self.adf = True
                self.result = adfuller(self.df)
                self.adf_stat = self.result[0]
                self.pval = self.result[1]
                self.critial_vals = self.result[4]
            if hurst:
                self.hurst = True
                H, c, data = hrst.compute_Hc(self.df)
                self.H = H
                self.c = c
                self.data = data
        def __repr__(self):
            result = ""
            if self.adf:
                result += 'Low ADF statistic tells us that the series is non-stationary.\n'
                result += 'P-values higher than 0.05 also indicate non-stationarity.\n'
                result += 'ADF Statistic: {}\n'.format(self.adf_stat)
                result += 'p-value: {}\n'.format(self.pval)
                result += 'Critial-Values: \n'
                for key, value in self.critial_vals.items():
                    result += "\t{}%: {}\n".format(key, value)
            if self.hurst:
                    result += 'H < 0.5 is stationary\nH > 0.05 is non-stationary\n'
                    result += 'H = 0.5 is random walk/Brownian motion.\n'
                    result += 'H = {:.4f}, c = {:.4f}'.format(self.H, self.c)
            return result

    class ModelPlotter:
        def __init__(self, df):
            self.fig = go.Figure()
            self.df = df
            for column in self.df.columns:
                self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='lines', name=column))

        def add_line(self, df, column):
            self.fig.add_trace(go.Scatter(x=self.df.index, y=df[column], mode='lines', name=column))

        def get_html_plot(self):
            plt_div = plot(self.fig, output_type='div')
            return plt_div

        def plot(self):
            self.fig.show()

    def __init__(self):
        self.assets = Asset.objects.all()

    def load_csv(self, path):
        df = pd.load_csv(os.path.join(BASE_DIR, path))
        df.index = df['date']
        self.multivariate_df = df
        return df

    def save(self, name):
        self.multivariate_df.to_csv(name)

    def load_db(self, label, set=False):
        try:
            label = Label.objects.get(model_name=label)
        except:
            raise Exception("Model does not exist.")
        columns = Column.objects.filter(label=label)
        queryset = Value.objects.filter(column__in=columns).order_by('date')
        q_df = pd.DataFrame(([q.column.name, q.date, q.value] for q in queryset), columns=['column', 'date', 'value'])
        column_list = list(q_df['column'].unique())
        df = pd.DataFrame(columns=column_list.append('date'))
        for column in q_df['column'].unique():
            cur_col = q_df.loc[q_df['column']==column]
            cur_col = cur_col.set_index('date')
            df[column] = cur_col['value']
        if set:
            self.multivariate_df = df
        return df


    def get_asset(self, asset=None, reset_df=True, start=None, end=None, column_name='return', zscore=True):
        unique = self.assets.values_list('ticker', flat=True)
        if asset==None:
            raise Exception('Not enough arguments provided, please pass a value for asset. The following are options for your model: \n{}'.format(unique))
        if not asset in unique:
            raise Exception('Asset not in dataset. Please provide one of the following options: {}'.format(unique))
        if not start:
            start = StockPrice.objects.filter(asset__ticker=asset).values_list('date').annotate(Min('date')).order_by('date')[0][1]
        if not end:
            end = StockPrice.objects.filter(asset__ticker=asset).values_list('date').annotate(Max('date')).order_by('-date')[0][1]

        stocks = StockPrice.objects.filter(asset__ticker=asset).filter(date__range=[start, end])
        asset_df = pd.DataFrame(([s.date, s.interday_volatility] for s in stocks), columns=['date', column_name]).set_index('date')
        if zscore:
            asset_df[column_name] = stats.zscore(asset_df[column_name])
        if reset_df:
            self.asset_name = asset
            self.dates = asset_df.index
            self.asset_df = asset_df
        return asset_df

    def get_sentiment(self, category='Negativ', set=False, zscore=True, column_name=None, source_filter=[], sentiment_words=[]):
        if category in SentimentPriceModel.sentiments:
            logging.info("Extracting {} sentiment from articles".format(category))
            sentiment_words = list(Category.objects.get(name=category).words.all().values_list('word', flat=True))

        elif len(sentiment_words) > 0:
            if category=='Negativ':
                raise Exception('Please specify a category name')
        else:
            raise Exception('Sentiment not specified. Please enter one of the options available by callings <SentimentPriceModelInstance>.sentiments or pass a list of sentiment words.')

        if not column_name:
            column_name = category

        if len(source_filter) > 0:
            sources = Source.objects.filter(id__in=source_filter).values_list('id', flat=True)
        else:
            sources = Source.objects.all().values_list('id', flat=True)

        sentiment_df = pd.DataFrame(([s['date'], s['sentiment']]
                                    for s in SentimentExtractor(self.dates,
                                                            sentiment_words,
                                                            sources)),
                                    columns=['date', column_name]).set_index('date')
            # get the z-score
        if zscore:
            sentiment_df[column_name] = stats.zscore(sentiment_df[column_name])
        if set:
            self.sentiment_df = sentiment_df
        return sentiment_df

    def test_stationary(self, column='return', adf=True, hurst=True, df=None):
        if not df:
            try:
                df = self.asset_df
            except:
                raise AttributeError('Please set the asset to test by calling get_asset()')
        stationary_results = self.StationaryResults(df, column, adf, hurst)
        print(stationary_results)
        return stationary_results

    def var_model(self, df=None, lag=1, freq=None):
        if df==None:
            df = self.multivariate_df
        if not freq==None:
            try:
                df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).to_period(freq)
            except:
                logging.info('Period type already set to {}'.format(df.index.freq))
        model = VAR(df)
        results = model.fit(lag)
        return results

    def add_asset_variable(self, left=None, asset=None, reset_df=False, set_multi=True, column_name='return', zscore=True):
        if not left:
            try:
                left = self.multivariate_df
            except:
                try:
                    left = self.asset_df
                except:
                    raise Exception('Please set the asset_df variable by calling get_asset().')
        if not asset:
            raise Exception('Not enough arguments provided, please pass a value for asset. The following are options for your model: \n{}'.format(self.asset_df['asset'].unique()))
        right = self.get_asset(asset=asset, reset_df=reset_df, column_name=column_name, zscore=zscore)
        merged = left.merge(right=right, on='date')
        if set_multi:
            self.multivariate_df = merged
        return merged

    def add_sentiment_variable(self, left=None, category=None, set=False, zscore=True, column_name=None, source_filter=[], reset_df=False, sentiment_words = []):
        if not left:
            try:
                left = self.multivariate_df
            except:
                try:
                    left = self.asset_df
                except:
                    raise Exception('Please set the asset_df variable by calling get_asset().')
        if not category:
            raise Exception('Please provide a sentiment category. Options can be found by calling <SentimentPriceModelInstance>.sentiments')
        if not column_name:
            column_name = right

        sentiment_df = self.get_sentiment(category=category, zscore=zscore, column_name=column_name, source_filter=source_filter, set=reset_df, sentiment_words=sentiment_words)
        new_df = left.merge(right=sentiment_df, on='date')
        if set:
            self.multivariate_df = new_df
        return sentiment_df

    def produce_plot_model(self, df=None):
        if df == None:
            try:
                df = self.multivariate_df
            except:
                try:
                    df = self.asset_df
                except:
                    try:
                        df = self.sentiment_df
                    except:
                        raise Exception("No dataframes set in the model.")
        self.mp = SentimentPriceModel.ModelPlotter(df=df)
        return self.mp



    def remove_variable(self, column=None):
        if not column:
            raise Exception("Please provide a column name.")
        try:
            del self.multivariate_df[column]
        except:
            try:
                del self.asset_df[column]
            except:
                raise Exception("Column does not exist in the object dataframe. Please try entering an existing column.")

    def save_to_database(self, name, description, df=None):
        if not df:
            try:
                df = self.multivariate_df
            except:
                try:
                    logging.info("Multivariate DF not set. Attempting to save asset DF.")
                    df = self.asset_df
                except:
                    raise Exception("Asset DF not set. Set this before attempting to save it to database.")
        object_creation_count = 0
        cols, rows = df.shape
        desired_count = cols*rows + len(df.columns) + 1
        label = Label.objects.create(model_name=name, description=description)
        label.save()
        object_creation_count +=1
        for col in df.columns:
            colobj = Column.objects.create(label=label, name=col)
            colobj.save()
            object_creation_count +=1
            df_col = df.loc[:, [col]]
            for index, row in df_col.iterrows():
                object_creation_count +=1
                progress(object_creation_count, desired_count, status="Saving column: {}".format(col))
                val = Value.objects.create(column=colobj, date=index, value=row[col])
                val.save()
        logging.info("Created {}/{} required objects during save".format(object_creation_count, desired_count))


class ArticleSentimentPriceModel(object):
    """
    Given a list of articles, the object finds their sentiment and the
    associated returns of markets on a given day.
    """
    class _SentiRet:
        def __init__(self, return, sentiment):
            self.returns = returns
            self.sentiment = sentiment

    def __init__(self, articles, asset, category):
        self.articles = articles
        words = 

        self.dates = list(set(articles.values_list('date_written', flat=True)))
        self.returns = StockPrice.objects.filter.filter(asset=asset).filter(date__in=self.dates).values_list('interday_volatility', flat=True)

    def __iter__(self):



def get_related_words(wv, word_list):
    dictionary = set()
    for word in word_list:
        if word in wv:
            [dictionary.add(x) for (x,y) in wv.most_similar(positive=[word], topn=5)]
            dictionary.add(word)
    new_word_list = list(dictionary)
    return new_word_list

def get_list(wv):
    neg = Category.objects.get(name="Negativ")
    neg_words = neg.words.all()

    pos = Category.objects.get(name="Positiv")
    pos_words = pos.words.all()

    neg_list = [word.word.lower() for word in neg_words]
    pos_list = [word.word.lower() for word in pos_words]

    neg_ = get_related_words(wv, neg_list)
    pos_ = get_related_words(wv, pos_list)

    subjective_words = neg_list + pos_list

    word_list = get_related_words(wv, subjective_words)

    return word_list, neg_, pos_


def clean_article(row):
    lines = []
    headline = row['headline']
    content = row['content']
    total = headline + ". " + content
    sentences = sent_tokenize(total)
    return sentences

def clean_articles_for_doc(row):
    total = row['headline'] + ". " + row["content"]
    return row

def convert_to_lines_df(input_file):
    df = pd.read_csv(input_file)
    df = df.loc[:, ['headline', 'content']]
    logging.info("Cleaning null values.")
    #
    logging.info(df.isnull().sum())
    df = df.dropna().reset_index(drop=True)
    logging.info(df.isnull().sum())
    #
    lines = []
    total, _ = df.shape
    for index, row in df.iterrows():
        sentences = clean_article(row)
        progress(index, total, status="Found {} sentences in article {}/{}".format(len(sentences), index, total))
        lines.extend(sentences)
    logging.info("Total lines found = {}".format(len(lines)))
    return lines

def mini_clean(text):
    return [re.sub("[^A-Za-z']+", ' ', str(text)).lower()]

def token_clean(doc):
    text = [token.lemma_ for token in doc if not token.is_stop]
    if len(text)>2:
        line = ' '.join(text)
        return line


def preprocess(lines):
    nlp = spacy.load("en_core_web_sm")
    docs = []
    t = time()
    index = 0
    mini_clean = (re.sub("[^A-Za-z']+", ' ', str(line)).lower() for line in lines)
    clean_lines = [token_clean(line) for line in nlp.pipe(mini_clean, disable=["tagger", "parser", "ner"])]
    print("Time to clean up everything: {}".format((time() - t)/ 60))
    return clean_lines

def phrase_detector(df):
    # create sentences
    sents = [line.split() for line in df['clean_docs']]
    phrases = Phrases(sents, min_count=80, progress_per=1000)
    bigram = Phraser(phrases)
    sentences = bigram[sents]
    logging.info("Finished replacing bigrams")
    logging.info(type(sentences))
    sentences.save("corpus.mm")
    return sentences


def create_model(sentences):
    w2v_model = gensim.models.Word2Vec(
                        min_count=20,
                        window=2,
                        size=120,
                        sample=2e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=multiprocessing.cpu_count()-1
                        )
    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.save("built_vocab.model")

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True)
    w2v_model.save("word2vec.model")
