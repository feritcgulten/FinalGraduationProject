from tkinter import font
from turtle import color
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash
from dash import Dash, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px 
import dash_html_components as html 
from datetime import datetime as dt
from datetime import date
import numpy as np
import pandas as pd
# to read and get data from file
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
from TurkishStemmer import TurkishStemmer
import matplotlib.gridspec as gridspec
# df = pd.read_json(r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\query_sma-ilac-saglik'
#                  r'_since_2020-01-01_tweets_22-08-47 22-03-2022.json')
# df.to_csv(r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\sma_ilac_saglik.csv', index=None)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import dash_bootstrap_components as dbc
string.punctuation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import defaultdict
stemmer = TurkishStemmer()
#Build your components 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'dbc.themes.BOOTSTRAP']

# df = pd.read_csv('sma_ilac_saglik.csv', encoding="utf-8").apply(lambda x: x.astype(str).str.lower())
# with utf-8 encoding the symbols for turkish characters are changed
# to observe raw data
def create_dashboard(flask_app):
    dash_app = dash.Dash(server=flask_app, name="Dashboard", url_base_pathname="/dash/",external_stylesheets=[dbc.themes.BOOTSTRAP])

  #Customize your own layout 
    dash_app.layout = html.Div([
        init_navbar(),
        html.Br(),
        dbc.Container([
        dbc.Row([dcc.DatePickerRange(
            id='my-date-picker-range',  # ID to be used for callback
            calendar_orientation='horizontal',  # vertical or horizontal
            day_size=39,  # size of calendar image. Default is 39
            end_date_placeholder_text="Return",  # text that appears when no end date chosen
            with_portal=False,  # if True calendar will open in a full screen overlay portal
            first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
            reopen_calendar_on_clear=True,
            is_RTL=False,  # True or False for direction of calendar
            clearable=True,  # whether or not the user can clear the dropdown
            number_of_months_shown=1,  # number of months shown when calendar is open
            min_date_allowed=dt(2020, 1, 1),  # minimum date allowed on the DatePickerRange component
            max_date_allowed=dt(2022, 3, 20),  # maximum date allowed on the DatePickerRange component
            initial_visible_month=dt(2020, 5, 1),  # the month initially presented when the user opens the calendar
            start_date=dt(2020, 1, 2).date(),
            end_date=dt(2022, 3, 15).date(),
            display_format='YYYY-MM-DD',  # how selected dates are displayed in the DatePickerRange component.
            month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.
            minimum_nights=2,  # minimum number of days between start and end date

            persistence=True,
            persisted_props=['start_date'],
            persistence_type='session',  # session, local, or memory. Default is 'local'

            updatemode='singledate'  # singledate or bothdates. Determines when callback is triggered
        )]),
        
        dbc.Row([html.H3("Top 10 Trend Topics", style={'textAlign': 'center'})
        ]),
        dbc.Row([dcc.Graph(id='mymap')
        ])
        ])
        
    ])
    

    @dash_app.callback(
        Output('mymap', 'figure'),
        [Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date')]
    )
    

    def index(start_date,end_date):
        sma = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\sma_ilac_saglik.csv'
        yogunbakim = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\yogun_bakim_yatak.csv'
        hastaneasi = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\hastane_asi_randevu.csv'
        hastaneentube = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\hastane_entube.csv'
        hastanepcr = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\hastane_pcr.csv'
        kanserilacibakanlik = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\kanser_ilaci_bakanlik.csv'
        smailacbakanlik = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\sma_ilac_bakanlik.csv'

        def concat(file):
            return pd.read_csv(file, encoding="utf-8").apply(lambda x: x.astype(str).str.lower())


        df = pd.concat(
            map(concat, [sma, hastaneasi, hastaneentube, hastanepcr, kanserilacibakanlik, smailacbakanlik]),
            ignore_index=True)

        df['created_at'] = pd.to_datetime(df['created_at'])
        df['created_at'] = df['created_at'].dt.date
        
        time = pd.date_range(start=start_date, end=end_date, freq="15D")
    # Filter data between two dates
        n = len(time)
        for i in range(0, n - 1, 1):

            filtered_df = df.loc[(df['created_at'] >= time[i])
                                & (df['created_at'] <= time[i + 1])]

            #filtered_df = df.loc[(df.created_at > start_date) & (df.created_at < end_date)] 
            print(filtered_df)
            if filtered_df['full_text'].count() < 2:
                print('Error')

            else:
                temp = filtered_df[['full_text', 'created_at']].values
                x = temp
                # x.shape
                # divide data as test and train applying n-gram
                (x_train, x_test) = train_test_split(x, test_size=0.5)
                # x_train.shape
                # x_test.shape
                x_train
                # assign the dataframes
                df1 = pd.DataFrame(x_train)
                df1 = df1.rename(columns={0: 'texts'})
                df_train = pd.concat([df1], axis=1)
                df_train.head()
                df_train.info()

                df3 = pd.DataFrame(x_test)
                df3 = df3.rename(columns={0: 'texts'})

                df_test = pd.concat([df3], axis=1)
                df_test.head()
                df_test.info()


                # remove punctuations
                def remove_punctuation(tweet):
                    if type(tweet) == float:
                        return tweet
                    ans = ""
                    for i in tweet:
                        if i not in string.punctuation:
                            ans += i
                    return ans


                # storing the punctuation free text in a new column
                df_train['texts'] = df_train['texts'].apply(lambda x: remove_punctuation(x))
                df_test['texts'] = df_test['texts'].apply(lambda x: remove_punctuation(x))
                df_train.head()
                stpwrd = nltk.corpus.stopwords.words('turkish')
                stop_list = ["sen", "ben", "bir", "insan", "siz", "onlar", "yap", " ", "yok", "var", "bug", "bun", "do", "1", "78",
                            "aşıs", "ko", "2", "bak", "lı", "da", "22", "", "an", "sade","https", "bur","mi"]
                stpwrd.extend(stop_list)
                for k in range(1, 5, 1):
                    # applying n-gram with function and removing stop words
                    # (pd.Series(nltk.ngrams(tweets, 2)).value_counts())[:10]
                    def generate_N_grams(text, ngram=k):
                        words_temp = [word for word in text.split(" ") if word not in set(stpwrd)]
                        words = [stemmer.stem(word) for word in words_temp]
                        words = list(map(lambda x: x.replace('bak', 'bakım'), words))
                        words = list(map(lambda x: x.replace('aş', 'aşı'), words))
                        words = list(map(lambda x: x.replace('oran', 'oranı'), words))
                        words = list(map(lambda x: x.replace('yok', 'yoğun'), words))
                        words = list(map(lambda x: x.replace('eriş', 'erişkin'), words))
                        words = list(map(lambda x: x.replace('erişkink', 'erişkin'), words))
                        words = list(map(lambda x: x.replace('sol', 'solunum'), words))
                        words = list(map(lambda x: x.replace('hast', 'hastane'), words))
                        words = list(map(lambda x: x.replace('bakımanlık', 'bakanlığı'), words))
                        words = list(map(lambda x: x.replace('hastaneane', 'hastane'), words))
                        words = list(map(lambda x: x.replace('hastanea', 'hastane'), words))
                        words = list(map(lambda x: x.replace('ülk', 'ülke'), words))
                        words = list(map(lambda x: x.replace('saglikbakımanlig', 'sağlıkbakanlığı'), words))
                        words = list(map(lambda x: x.replace('vak', 'vaka'), words))
                        words = list(map(lambda x: x.replace('bakıman', 'bakım'), words))
                        words = list(map(lambda x: x.replace('güçl', 'güçlü'), words))
                        words = list(map(lambda x: x.replace('sağlıkç', 'sağlıkçalışanı'), words))
                        words = list(map(lambda x: x.replace('drfahrettinkoc', 'drfahrettinkoca'), words))
                        words = list(map(lambda x: x.replace('bur', 'burda'), words))
                        words = list(map(lambda x: x.replace('ikinç', 'ikinci'), words))
                        words = list(map(lambda x: x.replace('sonr', 'sonra'), words))
                        temp = zip(*[words[i:] for i in range(0, ngram)])
                        ans = [' '.join(ngram) for ngram in temp]
                        return ans


                    allvalues = defaultdict(int)
                    # get the count of every word in both the columns of df_train and df_test dataframes for each sentiment type
                    for text in df_train.texts:
                        for word in generate_N_grams(text):
                            allvalues[word] += 1

                    df_positive = pd.DataFrame(sorted(allvalues.items(), key=lambda x: x[1], reverse=True))
                    if k == 1:
                        pd1 = df_positive[0][:10]
                        pd2 = df_positive[1][:10]
                    elif k == 2:
                        pd3 = df_positive[0][:10]
                        pd4 = df_positive[1][:10]
                    elif k == 3:
                        pd5 = df_positive[0][:10]
                        pd6 = df_positive[1][:10]
                    else:
                        pd7 = df_positive[0][:10]
                        pd8 = df_positive[1][:10]
                fig = make_subplots(rows=4, cols=1)
                fig.add_trace(go.Bar(x=pd2, y=pd1, name='unigram', orientation='h'),1,1)
                fig.add_trace(go.Bar(x=pd4, y=pd3, name='bigram', orientation='h'), 2, 1)
                fig.add_trace(go.Bar(x=pd6, y=pd5, name='trigram', orientation='h'), 3, 1)
                fig.add_trace(go.Bar(x=pd8, y=pd7, name='fourgram', orientation='h'), 4, 1)
                fig.update_layout(title_text=str(start_date) + ' -  ' + str(end_date), height=1500)
                return fig
    return dash_app      
    

def dashboards(flask_app):
    refresh_app = dash.Dash(server=flask_app, name="Newdata", url_base_pathname="/refresh/",external_stylesheets=[dbc.themes.BOOTSTRAP])
    refresh_app.layout = html.Div([
        init_navbar()
        ,
        html.Br(),
        dbc.Container([
        dbc.Row([dcc.DatePickerRange(
            id='my-date-picker-range2',  # ID to be used for callback
            calendar_orientation='horizontal',  # vertical or horizontal
            day_size=39,  # size of calendar image. Default is 39
            end_date_placeholder_text="Return",  # text that appears when no end date chosen
            with_portal=False,  # if True calendar will open in a full screen overlay portal
            first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
            reopen_calendar_on_clear=True,
            is_RTL=False,  # True or False for direction of calendar
            clearable=True,  # whether or not the user can clear the dropdown
            number_of_months_shown=1,  # number of months shown when calendar is open
            min_date_allowed=dt(2020, 1, 1),  # minimum date allowed on the DatePickerRange component
            max_date_allowed=dt(2022, 3, 20),  # maximum date allowed on the DatePickerRange component
            initial_visible_month=dt(2020, 5, 1),  # the month initially presented when the user opens the calendar
            start_date=dt(2020, 1, 2).date(),
            end_date=dt(2022, 3, 15).date(),
            display_format='YYYY-MM-DD',  # how selected dates are displayed in the DatePickerRange component.
            month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.
            minimum_nights=2,  # minimum number of days between start and end date

            persistence=True,
            persisted_props=['start_date'],
            persistence_type='session',  # session, local, or memory. Default is 'local'

            updatemode='singledate'  # singledate or bothdates. Determines when callback is triggered
        )]),
        
        dbc.Row([html.H3("New Top 10 Trend Topics", style={'textAlign': 'center'})
        ]),
        dbc.Row([dcc.Graph(id='mymap2')
        ])
        
        ])
    ])
    @refresh_app.callback(
        Output('mymap2', 'figure'),
        [Input('my-date-picker-range2', 'start_date'),
        Input('my-date-picker-range2', 'end_date')]
    )
    def new(start_date,end_date):
        sma = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\sma_ilac_saglik.csv'
        yogunbakim = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\yogun_bakim_yatak.csv'
        hastaneasi = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\hastane_asi_randevu.csv'
        hastaneentube = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\hastane_entube.csv'
        hastanepcr = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\hastane_pcr.csv'
        kanserilacibakanlik = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\kanser_ilaci_bakanlik.csv'
        smailacbakanlik = r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\sma_ilac_bakanlik.csv'
        yogunbakimyer =  r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\yogun_bakimda_yer.csv'
        def concat(file):
            return pd.read_csv(file, encoding="utf-8").apply(lambda x: x.astype(str).str.lower())


        df = pd.concat(
            map(concat, [sma, yogunbakim, hastaneasi, hastaneentube, hastanepcr, kanserilacibakanlik, smailacbakanlik, yogunbakimyer]),
            ignore_index=True)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['created_at'] = df['created_at'].dt.date
        print( df['created_at'])
      
        #filtered_df = df.loc[(df['created_at'] >= a)
        #             & (df['created_at'] < b)]
        print(start_date)
        #filtered_df =  df.loc[start_date:end_date]
        time = pd.date_range(start=start_date, end=end_date, freq="15D")
    # Filter data between two dates
        n = len(time)
        for i in range(0, n - 1, 1):

            filtered_df = df.loc[(df['created_at'] >= time[i])
                                & (df['created_at'] <= time[i + 1])]

            #filtered_df = df.loc[(df.created_at > start_date) & (df.created_at < end_date)] 
            print(filtered_df)
            if filtered_df['full_text'].count() < 2:
                print('Error')

            else:
                temp = filtered_df[['full_text', 'created_at']].values
                x = temp
                # x.shape
                # divide data as test and train applying n-gram
                (x_train, x_test) = train_test_split(x, test_size=0.5)
                # x_train.shape
                # x_test.shape
                x_train
                # assign the dataframes
                df1 = pd.DataFrame(x_train)
                df1 = df1.rename(columns={0: 'texts'})
                df_train = pd.concat([df1], axis=1)
                df_train.head()
                df_train.info()

                df3 = pd.DataFrame(x_test)
                df3 = df3.rename(columns={0: 'texts'})

                df_test = pd.concat([df3], axis=1)
                df_test.head()
                df_test.info()


                # remove punctuations
                def remove_punctuation(tweet):
                    if type(tweet) == float:
                        return tweet
                    ans = ""
                    for i in tweet:
                        if i not in string.punctuation:
                            ans += i
                    return ans


                # storing the punctuation free text in a new column
                df_train['texts'] = df_train['texts'].apply(lambda x: remove_punctuation(x))
                df_test['texts'] = df_test['texts'].apply(lambda x: remove_punctuation(x))
                df_train.head()
                stpwrd = nltk.corpus.stopwords.words('turkish')
                stop_list = ["sen", "ben", "bir", "insan", "siz", "onlar", "yap", " ", "yok", "var", "bug", "bun", "do", "1", "78",
                            "aşıs", "ko", "2", "bak", "lı", "da", "22", "", "an", "sade"]
                stpwrd.extend(stop_list)
                for k in range(1, 5, 1):
                    # applying n-gram with function and removing stop words
                    # (pd.Series(nltk.ngrams(tweets, 2)).value_counts())[:10]
                    def generate_N_grams(text, ngram=k):
                        words_temp = [word for word in text.split(" ") if word not in set(stpwrd)]
                        words = [stemmer.stem(word) for word in words_temp]
                        words = list(map(lambda x: x.replace('bak', 'bakım'), words))
                        words = list(map(lambda x: x.replace('aş', 'aşı'), words))
                        words = list(map(lambda x: x.replace('oran', 'oranı'), words))
                        words = list(map(lambda x: x.replace('yok', 'yoğun'), words))
                        words = list(map(lambda x: x.replace('eriş', 'erişkin'), words))
                        words = list(map(lambda x: x.replace('erişkink', 'erişkin'), words))
                        words = list(map(lambda x: x.replace('sol', 'solunum'), words))
                        words = list(map(lambda x: x.replace('hast', 'hastane'), words))
                        words = list(map(lambda x: x.replace('bakımanlık', 'bakanlığı'), words))
                        words = list(map(lambda x: x.replace('hastaneane', 'hastane'), words))
                        words = list(map(lambda x: x.replace('hastanea', 'hastane'), words))
                        words = list(map(lambda x: x.replace('ülk', 'ülke'), words))
                        temp = zip(*[words[i:] for i in range(0, ngram)])
                        ans = [' '.join(ngram) for ngram in temp]
                        return ans


                    allvalues = defaultdict(int)
                    # get the count of every word in both the columns of df_train and df_test dataframes for each sentiment type
                    for text in df_train.texts:
                        for word in generate_N_grams(text):
                            allvalues[word] += 1

                    df_positive = pd.DataFrame(sorted(allvalues.items(), key=lambda x: x[1], reverse=True))
                    if k == 1:
                        pd1 = df_positive[0][:10]
                        pd2 = df_positive[1][:10]
                    elif k == 2:
                        pd3 = df_positive[0][:10]
                        pd4 = df_positive[1][:10]
                    elif k == 3:
                        pd5 = df_positive[0][:10]
                        pd6 = df_positive[1][:10]
                    else:
                        pd7 = df_positive[0][:10]
                        pd8 = df_positive[1][:10]
                fig2 = make_subplots(rows=4, cols=1)
                fig2.add_trace(go.Bar(x=pd2, y=pd1, name='unigram', orientation='h'),1,1)
                fig2.add_trace(go.Bar(x=pd4, y=pd3, name='bigram', orientation='h'), 2, 1)
                fig2.add_trace(go.Bar(x=pd6, y=pd5, name='trigram', orientation='h'), 3, 1)
                fig2.add_trace(go.Bar(x=pd8, y=pd7, name='fourgram', orientation='h'), 4, 1)
                fig2.update_layout(title_text=str(start_date) + ' -  ' + str(end_date), height=1500)
                return fig2
    return refresh_app

def init_navbar():
    return dbc.NavbarSimple( className="navbar-dash",
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/", external_link=True)),
            dbc.NavItem(dbc.NavLink("Influencers", href="/influencers", external_link=True)),
            dbc.NavItem(dbc.NavLink("NewData", href="/refresh", external_link=True)),
        ],
        brand="Graduation Project",
        brand_href="/",
        brand_external_link=True,
        color="rgb(39, 147,248)",
        dark=True,
    )
