# Generated by Django 3.0.3 on 2020-03-05 18:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data_handler', '0017_article_contents'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='smarter_negative_words',
            field=models.IntegerField(null=True),
        ),
    ]
