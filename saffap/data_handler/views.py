from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.db.models import Count
from django.shortcuts import get_object_or_404
from django.db.models import Prefetch
from .preprocessing import produce_plots
from .data_handler import get_related_articles, get_relevant_stocks, produce_article_freq_plots
from rest_framework import generics
from data_handler.serializers import SourceSerializer, ArticleSerializer, WordCountSerializer, StockPriceSerializer, TimeSeriesSerializer, UniqueArticleSerializer
from data_handler.models import Source, Article, WordsInArticleCount, StockPrice, Asset
from sentiment.models import Category
from rest_pandas import PandasView
import pandas as pd
from sentiment.model import DocModelGenerator
from datetime import datetime, timedelta, date
import datetime as dt
from .forms import FilterForm, StockFilterForm
from .data_handler import get_dates, produce_stock_plots

class SourceViewSet(ListView):
    serializer_class = SourceSerializer
    template_name='data/sources.html'
    queryset = Source.objects.all()
    paginate_by=10

# Create your views here.
class ArticleView(ListView):
    template_name='data/articles.html'
    paginate_by = 10
    model = Article

    def get_queryset(self):
        self.form = FilterForm(data=self.request.GET or None)
        if self.request.GET and self.form.is_valid():
            request = self.request.GET

            self.date_start, self.date_end = get_dates(request)

            self.source_ids = request.getlist('source')
            self.queryset = Article.objects.filter(date_written__range=[self.date_start, self.date_end],
                                                source__id__in=self.source_ids)
        else:
            self.queryset = super().get_queryset()
        return self.queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['articles'] = self.queryset.count()
        context['plotdiv'] = produce_article_freq_plots(self.queryset)
        context['form'] = self.form
        return context

class ArticleDetailView(DetailView):
    template_name='data/articles_detail.html'
    model = Article

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        article = kwargs['object']
        dmg = DocModelGenerator()
        dmg.load_model("doc2vec.model")
        related = dmg.get_related_articles(article)
        related_stocks = get_relevant_stocks(related_articles)
        plot_div = produce_plots(related)

        context['related_articles'] = related_articles
        context['relevant_stocks'] = related_stocks
        context['plotdiv'] = plot_div
        return context

class WordCountView(ListView):
    paginate_by = 10
    serializer_class = WordCountSerializer
    template_name='data/wordcounts.html'
    def get_queryset(self):
        queryset = WordsInArticleCount.objects.filter(word__categories__name__in=["Negativ"]).values('word__word').annotate(total_articles=Count('word')).order_by('-total_articles')
        return queryset

class CategoryView(ListView):
    paginate_by = 10
    template_name = 'data/categories.html'
    queryset = Category.objects.all()


class FinancialDataView(ListView):
    paginate_by = 10
    template_name='data/stockprices.html'
    queryset = StockPrice.objects.all().order_by('date')
    model = StockPrice
    def get_queryset(self):
        # asset = Asset.objects.filter(name="FTSE 100").first()
        self.form = StockFilterForm(data=self.request.GET or None)
        print(self.request.GET)
        if self.request.GET and self.form.is_valid():
            request = self.request.GET

            self.date_start, self.date_end = get_dates(request)

            self.asset_ids = request.getlist('assets')
            if not self.asset_ids:
                self.asset_ids = [asset.id for asset in (Asset.objects.all())]
            self.queryset = StockPrice.objects.filter(date__range=[self.date_start, self.date_end],
                                                asset__id__in=self.asset_ids)
        else:
            self.queryset = super().get_queryset()
        return self.queryset

    def get_context_data(self, **kwargs):
            context = super().get_context_data(**kwargs)
            context['plotdiv'] = produce_stock_plots(self.queryset)
            context['stocks'] = self.queryset.count()
            context['form'] = self.form
            return context
