from .models import Article, StockPrice
from gensim.models import Doc2Vec
from django.conf import settings
from django.db.models import Count
from plotly.offline import plot
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import numpy as np
from django.conf import settings
from scipy import stats

def produce_aggregate_plots(start, end):
    # get article sentiment by summing neg words over the course of the day and diving by length of words over the day
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    index = 0
    df = pd.DataFrame(([p.date, p.pos_sum, p.neg_sum,
                        p.length, p.neg_words, p.interday_volatility,
                        p.asset, p.pos_sent, p.neg_sent,
                        p.naive_sent]
                        for p in AggregateRowIterator(start, end)),
                        columns=['date', 'pos_sum', 'neg_sum',
                                'length', 'neg_words', 'return',
                                'asset', 'pos_sent', 'neg_sent',
                                'naive_sent'])
    fig = go.Figure()
    dates = df['date'].drop_duplicates().to_frame()
    neg_sent = df['neg_sent'].drop_duplicates().to_frame()
    pos_sent = df['pos_sent'].drop_duplicates().to_frame()

    gbpeur = df.loc[df['asset']=="GBP/EUR"]
    gbpusd = df.loc[df['asset']=="GBP/USD"]

    ftse100 = df.loc[df['asset']=="FTSE 100"]
    ftse100['return'] = ftse100['return'].apply(lambda x: x/100)

    fig.add_trace(go.Scatter(x=dates['date'], y=neg_sent['neg_sent'], mode='lines', name='Negative Sentiment'))
    fig.add_trace(go.Scatter(x=dates['date'], y=pos_sent['pos_sent'], mode='lines', name='Positive Sentiment'))
    fig.add_trace(go.Scatter(x=gbpeur['date'], y=gbpeur['return'], mode='lines', name='GBP/EUR'))
    fig.add_trace(go.Scatter(x=gbpusd['date'], y=gbpusd['return'], mode='lines', name='GBP/USD'))
    fig.add_trace(go.Scatter(x=ftse100['date'], y=ftse100['return'], mode='lines', name='FTSE 100'))

    plt_div = plot(fig, output_type='div')
    return plt_div

def produce_article_plots(articles):
    length = len(articles)
    df = pd.DataFrame(([p.date, p.pos_sent, p.neg_sent,
                        p.interday_volatility, p.asset]
                        for p in RowIterator(articles)),
                        columns=['date', 'pos_sent', 'neg_sent',
                                'return', 'asset'])
    print(df)
    df = df.sort_values(by=['date'])
    fig = go.Figure()

    dates = df['dates'].drop_duplicates().to_frame()
    neg_sent = df['neg_sent']

    fig = px.line(df, x="date", y="value", color='line', line_group='line')
    plt_div = plot(fig, output_type='div')
    return df


def produce_table(start, end):
    df = collect_data(start, end)
    return df

def produce_article_freq_plots(queryset):
    articles = queryset.values('date_written').annotate(total=Count('date_written')).order_by('-total')
    df = pd.DataFrame(([a['date_written'], a['total']] for a in articles), columns=['date', 'count'])
    df = df.sort_values(by=['date'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['count'], mode='lines', name='Articles per Day'))
    plt_div = plot(fig, output_type='div')
    return plt_div

def produce_stock_plots(queryset):
    df = pd.DataFrame(([s.date, s.asset.name, s.asset.ticker, s.open, s.close, s.interday_volatility] for s in queryset),
                        columns=['date', 'name', 'ticker', 'open', 'close', 'return'])
    unique_assets = df.ticker.unique()
    df = df.sort_values(by=['date'])
    df = df.drop_duplicates()
    fig = go.Figure()
    for asset in unique_assets:
        line = df.loc[df['ticker']==asset]
        avg_return = stats.zscore(df['return'])
        fig.add_trace(go.Scatter(x=line['date'], y=avg_return, mode='lines', name=asset))
    plt_div = plot(fig, output_type='div')
    return plt_div

def get_dates(request):
    start_day = int(request.get('date_start_day'))
    start_month = int(request.get('date_start_month'))
    start_year = int(request.get('date_start_year'))

    end_day = int(request.get('date_end_day'))
    end_month = int(request.get('date_end_month'))
    end_year = int(request.get('date_end_year'))

    date_start = datetime(day=start_day, month=start_month, year=start_year).date()
    date_end = datetime(day=end_day, month=end_month, year=end_year).date()

    return date_start, date_end


def get_related_articles(article):
    dmodel = Doc2Vec.load(settings.DOC2VECMODEL)
    similar = dmodel.docvecs.most_similar(article.headline, topn=20)
    articles = []
    for headline, rating in similar:
        cur = Article.objects.get(headline=headline)
        articles.append([cur, rating])
    return articles

def get_relevant_stocks(articles):
    stocks = []
    for article, rating in articles:
        cur = StockPrice.objects.filter(date=article.date_written)
        if cur:
            stocks.append(cur)
    return stocks
