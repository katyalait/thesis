from django.shortcuts import render
from django.db.models import Count
from .aggregator import produce_table, get_article_sentiments, produce_plots
from . import models
from rest_framework import generics
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from django.views.generic.base import TemplateView
from data_handler.models import Asset, Article, Word
from sentiment.models import Category
from datetime import datetime, date
from .forms import FilterForm
from sentiment.model import SentimentPriceModel
from data_handler.data_handler import get_dates
# Create your views here.

class Sentiment(TemplateView):
    template_name = 'sentiment/index.html'

class GraphSentiment(TemplateView):
    model_name = "full_monty2"
    template_name = 'sentiment/graph.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.form = FilterForm(data=self.request.GET or None)
        context['form'] = self.form
        spm = SentimentPriceModel()
        df = spm.load_db(label=self.model_name, set=True)
        if self.request.GET and self.form.is_valid():
            request = self.request.GET
            date_start, date_end = get_dates(request)
            print(dates)
            asset_ids = request.getlist('assets')
            print("Asset ids: {}".format(asset_ids))
            tickers = Asset.objects.filter(id__in=asset_ids).values_list('ticker', flat=True)
            df = df.loc[:, tickers]
            df = df.loc[date_start:date_end]
            mp = spm.produce_plot_model(df=df)
            context['plotdiv'] = mp.get_html_plot()
            #context['plotdiv'] = produce_plots(date_start, date_end, assets)
        else:
            mp = spm.produce_plot_model(df=df)
            context['plotdiv'] = mp.get_html_plot()
            # context['plotdiv'] = produce_plots(date_start, date_end, assets)
        return context

class TableSentiment(TemplateView):
    # changed template
    template_name = 'sentiment/table.html'
    paginate_by = 10
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['table'] = produce_table('2016-01-01','2016-12-31')
        return context
