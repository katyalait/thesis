from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    path('view_sentiment', views.Sentiment.as_view(), name="sentiment"),
    path('view_table', views.TableSentiment.as_view(), name="table_sent"),
    path('view_graph', views.GraphSentiment.as_view(), name="graph_sent"),

]
urlpatterns = format_suffix_patterns(urlpatterns)
