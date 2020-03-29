from data_handler import models
from rest_framework import serializers

class UniqueArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Article
        fields = ['date_written']

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Article
        fields = ['headline', 'source', 'length', 'date_written']

class WordCountSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.WordsInArticleCount
        fields = ['article', 'word', 'frequency']

class StockPriceSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.StockPrice
        fields = '__all__'

class TimeSeriesSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.StockPrice
        fields = ['date', 'interday_volatility']

class SourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Source
        fields = '__all__'
