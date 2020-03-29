from django import forms
from data_handler.models import Asset
from datetime import date

class FilterForm(forms.Form):
    year_range = tuple([i for i in range(2016, 2021)])
    assets = Asset.objects.all()
    date_start = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    date_end = forms.DateField(widget=forms.SelectDateWidget(years=year_range))
    assets = forms.MultipleChoiceField(choices=[(asset.id, asset.name) for asset in assets], widget=forms.CheckboxSelectMultiple())

    def __init__(self, *args, **kwargs):
        super(FilterForm, self).__init__(*args, **kwargs)
