---
title:  "NLP Project - Analyzing Covid Vaccine Tweets Sentiment (Python, GeoPy, Vader-sentiment, PorterStemmer)"

tagline: " "
header:
  overlay_image: /assets/vaccine-post/tweet.jpg
  caption: "Photo credit: [**MDGovpics**](https://search.creativecommons.org/photos/01fdd347-e40b-4a70-9634-08713a49e7f8)"
---

This page contains an easily readable report on the data analysis. To view the code, click here:  
## [Link to Jupyter notebook on GitHub](https://github.com/ain237/data-science-projects/blob/main/nlp-vaccine-tweets/vaccine_tweets.ipynb " ")


## Introduction


The data is a collection of tweets retrieved from Kaggle. The tweets were collected by an individual using the Twitter API, using hashtags to pick tweets containing information about the Pfizer & BioNTech Vaccine.

Link to the original data: https://www.kaggle.com/gpreda/pfizer-vaccine-tweets (In this notebook: Version 119)

The data is around 9000 entries, some of which are unusable for our analysis. While there are bigger datasets with hundreds of thousands of rows available, this dataset was selected because of it's good quality, and the ability to run GeoPy to make inferences about locations.

Most of the cells in this project are made intentionally to combine code as much as possible into one cell. This is to increase readability on the portfolio website where this is posted!

Hopefully you'll enjoy reading this NLP project.


**Table of contents:**
- The Data: Exploratory analysis
    - Data Cleanup
    - Converting time strings to datetime format
- Sentiment Analysis
    - Using Vader-score to understand the sentiment
    - Calculating the followers to following ratio.
    - Regression analysis
- Text cleanup using PorterStemmer and plotting the common words.
- Conclusions



```python
#Load the libraries

import numpy as np
import pandas as pd
import math
import random

from datetime import datetime

#!pip3 install geopy

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from sklearn.feature_extraction.text import CountVectorizer
import collections

import re, string, os

import nltk
nltk.download('stopwords')
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import matplotlib.dates as mdates

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#!pip3 install researchpy
import researchpy as rp
import scipy.stats as stats


#Load the data
tweet_data_original = pd.read_csv('vaccination_tweets.csv')
```

    [nltk_data] Downloading package stopwords to /Users/tatu/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     /Users/tatu/nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!


## The Data: Exploratory analysis

To understand the data, we will perform several checks and cleanup to get the data we want for later analysis. The data needs to be also reviewed in order to choose the types of analysis we can conduct on it.






```python
tweet_data_original.head()[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_name</th>
      <th>user_location</th>
      <th>user_description</th>
      <th>user_created</th>
      <th>user_followers</th>
      <th>user_friends</th>
      <th>user_favourites</th>
      <th>user_verified</th>
      <th>date</th>
      <th>text</th>
      <th>hashtags</th>
      <th>source</th>
      <th>retweets</th>
      <th>favorites</th>
      <th>is_retweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1340539111971516416</td>
      <td>Rachel Roh</td>
      <td>La Crescenta-Montrose, CA</td>
      <td>Aggregator of Asian American news; scanning di...</td>
      <td>2009-04-08 17:52:46</td>
      <td>405</td>
      <td>1692</td>
      <td>3247</td>
      <td>False</td>
      <td>2020-12-20 06:06:44</td>
      <td>Same folks said daikon paste could treat a cyt...</td>
      <td>['PfizerBioNTech']</td>
      <td>Twitter for Android</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1338158543359250433</td>
      <td>Albert Fong</td>
      <td>San Francisco, CA</td>
      <td>Marketing dude, tech geek, heavy metal &amp; '80s ...</td>
      <td>2009-09-21 15:27:30</td>
      <td>834</td>
      <td>666</td>
      <td>178</td>
      <td>False</td>
      <td>2020-12-13 16:27:13</td>
      <td>While the world has been on the wrong side of ...</td>
      <td>NaN</td>
      <td>Twitter Web App</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweet_data_original.info()
tweet_data_original.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9541 entries, 0 to 9540
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   id                9541 non-null   int64 
     1   user_name         9541 non-null   object
     2   user_location     7623 non-null   object
     3   user_description  8944 non-null   object
     4   user_created      9541 non-null   object
     5   user_followers    9541 non-null   int64 
     6   user_friends      9541 non-null   int64 
     7   user_favourites   9541 non-null   int64 
     8   user_verified     9541 non-null   bool  
     9   date              9541 non-null   object
     10  text              9541 non-null   object
     11  hashtags          7295 non-null   object
     12  source            9540 non-null   object
     13  retweets          9541 non-null   int64 
     14  favorites         9541 non-null   int64 
     15  is_retweet        9541 non-null   bool  
    dtypes: bool(2), int64(6), object(8)
    memory usage: 1.0+ MB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_followers</th>
      <th>user_friends</th>
      <th>user_favourites</th>
      <th>retweets</th>
      <th>favorites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.541000e+03</td>
      <td>9.541000e+03</td>
      <td>9541.000000</td>
      <td>9.541000e+03</td>
      <td>9541.000000</td>
      <td>9541.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.362976e+18</td>
      <td>3.618268e+04</td>
      <td>1177.470705</td>
      <td>1.481357e+04</td>
      <td>1.425532</td>
      <td>8.176606</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.244526e+16</td>
      <td>3.100293e+05</td>
      <td>2846.392349</td>
      <td>4.643485e+04</td>
      <td>12.087367</td>
      <td>54.859691</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.337728e+18</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.346552e+18</td>
      <td>1.100000e+02</td>
      <td>165.000000</td>
      <td>4.110000e+02</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.354802e+18</td>
      <td>4.690000e+02</td>
      <td>462.000000</td>
      <td>2.300000e+03</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.373953e+18</td>
      <td>2.098000e+03</td>
      <td>1228.000000</td>
      <td>1.120600e+04</td>
      <td>1.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.423141e+18</td>
      <td>1.371493e+07</td>
      <td>103226.000000</td>
      <td>1.166459e+06</td>
      <td>678.000000</td>
      <td>2315.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweet_data_original['user_location'].value_counts()
```




    Malaysia                 151
    London, England          135
    India                    119
    London                    93
    Canada                    86
                            ... 
    Bagni Di Lucca, Italy      1
    Honolulu Africa/Asia       1
    Tokyo/ Pune                1
    NHS legal advisor          1
    London (UK)                1
    Name: user_location, Length: 2775, dtype: int64



## Data Cleanup


```python
#Checking the instesting columns for null values and dropping all rows with null values

tweet_data = tweet_data_original[tweet_data_original['user_location'].notna()]
tweet_data = tweet_data[tweet_data['user_description'].notna()]

#Choosing only interesting columns
tweet_data = tweet_data[['user_location','user_description','user_followers','user_friends','user_verified','date','text']]


tweet_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7353 entries, 0 to 9540
    Data columns (total 7 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   user_location     7353 non-null   object
     1   user_description  7353 non-null   object
     2   user_followers    7353 non-null   int64 
     3   user_friends      7353 non-null   int64 
     4   user_verified     7353 non-null   bool  
     5   date              7353 non-null   object
     6   text              7353 non-null   object
    dtypes: bool(1), int64(2), object(4)
    memory usage: 409.3+ KB


### Using GeoPy to convert locations to country locations

#### This script takes in fact around 1.5 hours to run. It checks for a csv file in the project folder, which if found, will enable the script to skip the data-collection and speed up the running of the notebook.


```python
geopy = RateLimiter(Nominatim(user_agent='xccyyy').geocode, min_delay_seconds=1)

geo_df=pd.DataFrame(columns=['Latitude','Longitude','Country'])

try:
    csv_df=pd.read_csv('geo_df.csv')
    print('csv found')
    print(len(csv_df))
except:
    print('No csv, continuing.')
    csv_df=pd.DataFrame(columns=['Latitude','Longitude','Country'])

loc_list_lat = []
loc_list_lon = []
loc_list_countries = []
counter = 0
found_counter = 0

if len(csv_df) < 6500:
    for location in tweet_data['user_location'].tolist():
    
        counter = counter + 1

        if counter == 10:
            print('10 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 500:
            print('11 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 1000:
            print('1000 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 1500:
            print('1500 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 2000:
            print('2000 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 3000:
            print('3000 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 4000:
            print('4000 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 5000:
            print('5000 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 6000:
            print('6000 done')
            print('Found: ' + str(found_counter))
            print(' ')
        elif counter == 7200:
            print('almost done')

        try:
            place = geopy(location)
            loc_list_lat.append(place.raw['lat'])
            loc_list_lon.append(place.raw['lon'])
            loc_list_countries.append(place.raw['display_name'].split(", ")[-1])
            found_counter = found_counter + 1
        except:
            loc_list_lat.append('None')
            loc_list_lon.append('None')
            loc_list_countries.append('None')

        geo_df['Latitude'] = loc_list_lat
        geo_df['Longitude'] = loc_list_lon
        geo_df['Country'] = loc_list_countries
        geo_df.to_csv('geo_df.csv')

geo_df=pd.read_csv('geo_df.csv')
geo_df.head()

```

    csv found
    7353





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>34.2192742</td>
      <td>-118.2318871</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>37.7790262</td>
      <td>-122.419906</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>42.6851664</td>
      <td>-2.942752</td>
      <td>España</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>49.2608724</td>
      <td>-123.1139529</td>
      <td>Canada</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>52.4796992</td>
      <td>-1.9026911</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
geo_df['Country'].value_counts()[0:10]
```




    United States               1535
    United Kingdom              1288
    None                         908
    Canada                       517
    India                        459
    Malaysia                     247
    الإمارات العربية المتحدة     207
    Éire / Ireland               196
    Deutschland                  168
    France                       132
    Name: Country, dtype: int64



## For our analysis we will choose the countries United States, United Kingdom, Canada, India and Malaysia because they have sufficiently large amount of tweets for analysis.


```python
#adding the geo_df to the tweets dataframe

location_data = pd.read_csv('geo_df.csv')
tweet_data = tweet_data[['user_description','user_followers','user_friends','user_verified','date','text']]
tweet_data['Country'] = location_data['Country']
tweet_data['Latitude'] = location_data['Latitude']
tweet_data['Longitude'] = location_data['Longitude']

tweet_data.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_description</th>
      <th>user_followers</th>
      <th>user_friends</th>
      <th>user_verified</th>
      <th>date</th>
      <th>text</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aggregator of Asian American news; scanning di...</td>
      <td>405</td>
      <td>1692</td>
      <td>False</td>
      <td>2020-12-20 06:06:44</td>
      <td>Same folks said daikon paste could treat a cyt...</td>
      <td>United States</td>
      <td>34.2192742</td>
      <td>-118.2318871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marketing dude, tech geek, heavy metal &amp; '80s ...</td>
      <td>834</td>
      <td>666</td>
      <td>False</td>
      <td>2020-12-13 16:27:13</td>
      <td>While the world has been on the wrong side of ...</td>
      <td>United States</td>
      <td>37.7790262</td>
      <td>-122.419906</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heil, hydra 🖐☺</td>
      <td>10</td>
      <td>88</td>
      <td>False</td>
      <td>2020-12-12 20:33:45</td>
      <td>#coronavirus #SputnikV #AstraZeneca #PfizerBio...</td>
      <td>España</td>
      <td>42.6851664</td>
      <td>-2.942752</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hosting "CharlesAdlerTonight" Global News Radi...</td>
      <td>49165</td>
      <td>3933</td>
      <td>True</td>
      <td>2020-12-12 20:23:59</td>
      <td>Facts are immutable, Senator, even when you're...</td>
      <td>Canada</td>
      <td>49.2608724</td>
      <td>-123.1139529</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gastroenterology trainee, Clinical Research Fe...</td>
      <td>105</td>
      <td>108</td>
      <td>False</td>
      <td>2020-12-12 20:11:42</td>
      <td>Does anyone have any useful advice/guidance fo...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



### Converting time strings to datetime format


```python
datelist = []
for date in tweet_data['date'].tolist():
    datelist.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
tweet_data['datetime'] = datelist
del tweet_data['date']
tweet_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_description</th>
      <th>user_followers</th>
      <th>user_friends</th>
      <th>user_verified</th>
      <th>text</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aggregator of Asian American news; scanning di...</td>
      <td>405</td>
      <td>1692</td>
      <td>False</td>
      <td>Same folks said daikon paste could treat a cyt...</td>
      <td>United States</td>
      <td>34.2192742</td>
      <td>-118.2318871</td>
      <td>2020-12-20 06:06:44</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marketing dude, tech geek, heavy metal &amp; '80s ...</td>
      <td>834</td>
      <td>666</td>
      <td>False</td>
      <td>While the world has been on the wrong side of ...</td>
      <td>United States</td>
      <td>37.7790262</td>
      <td>-122.419906</td>
      <td>2020-12-13 16:27:13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heil, hydra 🖐☺</td>
      <td>10</td>
      <td>88</td>
      <td>False</td>
      <td>#coronavirus #SputnikV #AstraZeneca #PfizerBio...</td>
      <td>España</td>
      <td>42.6851664</td>
      <td>-2.942752</td>
      <td>2020-12-12 20:33:45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hosting "CharlesAdlerTonight" Global News Radi...</td>
      <td>49165</td>
      <td>3933</td>
      <td>True</td>
      <td>Facts are immutable, Senator, even when you're...</td>
      <td>Canada</td>
      <td>49.2608724</td>
      <td>-123.1139529</td>
      <td>2020-12-12 20:23:59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gastroenterology trainee, Clinical Research Fe...</td>
      <td>105</td>
      <td>108</td>
      <td>False</td>
      <td>Does anyone have any useful advice/guidance fo...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2020-12-12 20:11:42</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Double checking that all the needed data is in correct format!

tweet_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7353 entries, 0 to 9540
    Data columns (total 9 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   user_description  7353 non-null   object        
     1   user_followers    7353 non-null   int64         
     2   user_friends      7353 non-null   int64         
     3   user_verified     7353 non-null   bool          
     4   text              7353 non-null   object        
     5   Country           5654 non-null   object        
     6   Latitude          5654 non-null   object        
     7   Longitude         5654 non-null   object        
     8   datetime          7353 non-null   datetime64[ns]
    dtypes: bool(1), datetime64[ns](1), int64(2), object(5)
    memory usage: 524.2+ KB


## Sentiment Analysis

### Using Vader-score to understand the sentiment


```python

```


```python
#Inspired by several online tutorials, sources missing.

vader = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):
    snt = vader.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(snt)))

#Seeing if Vader works
print_sentiment_scores("Vaccine bad") #Compound value scale = -1 to 1 (-ve to +ve)
```

    Vaccine bad----------------------------- {'neg': 0.778, 'neu': 0.222, 'pos': 0.0, 'compound': -0.5423}



```python
def vader_to_list(df,column):
    %time   #to calulate the time it takes the algorithm to compute a VADER score
    i=0
    to_list = [ ] 
    while (i<len(df)):
    
        k = vader.polarity_scores(df.iloc[i][column])
        to_list.append(k['compound'])
        
        i = i+1
        
    return to_list

tweet_data['VADER tweet'] = vader_to_list(tweet_data,'text')
tweet_data['VADER descr'] = vader_to_list(tweet_data,'user_description')

tweet_data.head()
```

    CPU times: user 2 µs, sys: 1 µs, total: 3 µs
    Wall time: 6.2 µs
    CPU times: user 2 µs, sys: 0 ns, total: 2 µs
    Wall time: 3.81 µs





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_description</th>
      <th>user_followers</th>
      <th>user_friends</th>
      <th>user_verified</th>
      <th>text</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>datetime</th>
      <th>VADER tweet</th>
      <th>VADER descr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aggregator of Asian American news; scanning di...</td>
      <td>405</td>
      <td>1692</td>
      <td>False</td>
      <td>Same folks said daikon paste could treat a cyt...</td>
      <td>United States</td>
      <td>34.2192742</td>
      <td>-118.2318871</td>
      <td>2020-12-20 06:06:44</td>
      <td>0.4019</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marketing dude, tech geek, heavy metal &amp; '80s ...</td>
      <td>834</td>
      <td>666</td>
      <td>False</td>
      <td>While the world has been on the wrong side of ...</td>
      <td>United States</td>
      <td>37.7790262</td>
      <td>-122.419906</td>
      <td>2020-12-13 16:27:13</td>
      <td>-0.1027</td>
      <td>0.3182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heil, hydra 🖐☺</td>
      <td>10</td>
      <td>88</td>
      <td>False</td>
      <td>#coronavirus #SputnikV #AstraZeneca #PfizerBio...</td>
      <td>España</td>
      <td>42.6851664</td>
      <td>-2.942752</td>
      <td>2020-12-12 20:33:45</td>
      <td>0.2500</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hosting "CharlesAdlerTonight" Global News Radi...</td>
      <td>49165</td>
      <td>3933</td>
      <td>True</td>
      <td>Facts are immutable, Senator, even when you're...</td>
      <td>Canada</td>
      <td>49.2608724</td>
      <td>-123.1139529</td>
      <td>2020-12-12 20:23:59</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gastroenterology trainee, Clinical Research Fe...</td>
      <td>105</td>
      <td>108</td>
      <td>False</td>
      <td>Does anyone have any useful advice/guidance fo...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2020-12-12 20:11:42</td>
      <td>0.7003</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Aanalysis: Difference in sentiment between different countries.
```


```python
country_comparison=pd.DataFrame(columns=['Country','Vader score'])
country_list = ['United States', 'United Kingdom', 'Canada', 'India', 'Malaysia']
vader_list = []
vader_list.append(tweet_data.loc[tweet_data['Country'] == 'United States']['VADER tweet'].mean())
vader_list.append(tweet_data.loc[tweet_data['Country'] == 'United Kingdom']['VADER tweet'].mean())
vader_list.append(tweet_data.loc[tweet_data['Country'] == 'Canada']['VADER tweet'].mean())
vader_list.append(tweet_data.loc[tweet_data['Country'] == 'India']['VADER tweet'].mean())
vader_list.append(tweet_data.loc[tweet_data['Country'] == 'Malaysia']['VADER tweet'].mean())
country_comparison['Country'] = country_list
country_comparison['Vader score'] = vader_list

#country_comparison.sort_values(by=['Vader score'])

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.set(font_scale=0.85)

comparison_graph = sns.barplot(x='Country', y='Vader score', palette="PuBuGn_d", data=country_comparison.sort_values(by=['Vader score'])).set(xlabel=None)
                         
plt.show()





                  

```


    
![output_22_0]({{"assets/vaccine-post/output_22_0.png" | relative_url }} " ")



### Canadians seem to be the most positive about the vaccine, while Indians were the most negative in this dataset.

### User analysis: How does the sentiment differ between verified and non-verified users?



```python
sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
verified = sns.barplot(x="user_verified", y="VADER tweet", data=tweet_data, estimator=np.mean, ci=85, capsize=.2, palette="PuBuGn_d")

plt.show()


```


    
![output_25_0]({{"assets/vaccine-post/output_25_0.png" | relative_url }} " ")



### The results seem odd. Testing with T-Test to confirm the result.


```python
rp.ttest(group1= tweet_data['VADER tweet'][tweet_data['user_verified'] == True], group1_name= "Verified",
         group2= tweet_data['VADER tweet'][tweet_data['user_verified'] == False], group2_name= "Non-verified")
```




    (       Variable       N      Mean        SD        SE  95% Conf.  Interval
     0      Verified   789.0  0.078502  0.306532  0.010913   0.057080  0.099923
     1  Non-verified  6564.0  0.140601  0.372281  0.004595   0.131594  0.149609
     2      combined  7353.0  0.133938  0.366278  0.004271   0.125565  0.142311,
                             Independent t-test    results
     0  Difference (Verified - Non-verified) =     -0.0621
     1                    Degrees of freedom =   7351.0000
     2                                     t =     -4.5054
     3                 Two side test p value =      0.0000
     4                Difference < 0 p value =      0.0000
     5                Difference > 0 p value =      1.0000
     6                             Cohen's d =     -0.1698
     7                             Hedge's g =     -0.1697
     8                         Glass's delta =     -0.2026
     9                           Pearson's r =      0.0525)



## The verified users tend to indeed tweet more negative things in this dataset. Maybe we can find an explanation for this...


## Let's see some of these tweets!


```python
verifieds = (tweet_data[['text', 'VADER tweet']][tweet_data['user_verified'] == True])

#verifieds.head()
verifieds.loc[verifieds['VADER tweet'] < -0.2]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>VADER tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>#FDA authorizes #PfizerBioNTech #coronavirus v...</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Turkey’s coronavirus death toll reaches 16,417...</td>
      <td>-0.5719</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Turkey’s coronavirus death toll reaches 16,417...</td>
      <td>-0.5719</td>
    </tr>
    <tr>
      <th>235</th>
      <td>The US Food and Drug Administration (FDA) issu...</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Global coronavirus death toll reaches 1,611,63...</td>
      <td>-0.5719</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8328</th>
      <td>MoHAP Authorised the Emergency Use of the Pfiz...</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>8528</th>
      <td>Can @pfizer, @moderna_tx vaccines be gamechang...</td>
      <td>-0.6249</td>
    </tr>
    <tr>
      <th>8801</th>
      <td>Deliveries of the vaccine, among several brand...</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>8860</th>
      <td>#NSTworld #EU states must use all the #vaccine...</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>9490</th>
      <td>Israel will roll out a booster dose of the #Pf...</td>
      <td>-0.2732</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



## Many of the negative tweets from the verified accounts seem to be news stories that are not negative about the vaccine itself. It might be more useful to focus our analysis on regular users, who have close to 1:1 followers to following ratio...


### Calculating the followers to following ratio.


```python
#The next function ran into trouble because there were infinite values because of division by zero, let's see why.
(tweet_data.user_friends == 0).sum()

#There seems to be 27 zero values in the dataset. This is not too much to just remove.
popularity_data = tweet_data[tweet_data.user_friends != 0]
popularity_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7326 entries, 0 to 9540
    Data columns (total 11 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   user_description  7326 non-null   object        
     1   user_followers    7326 non-null   int64         
     2   user_friends      7326 non-null   int64         
     3   user_verified     7326 non-null   bool          
     4   text              7326 non-null   object        
     5   Country           5633 non-null   object        
     6   Latitude          5633 non-null   object        
     7   Longitude         5633 non-null   object        
     8   datetime          7326 non-null   datetime64[ns]
     9   VADER tweet       7326 non-null   float64       
     10  VADER descr       7326 non-null   float64       
    dtypes: bool(1), datetime64[ns](1), float64(2), int64(2), object(5)
    memory usage: 636.7+ KB



```python
popularity_data['followers_to_following'] = (popularity_data['user_followers'] / popularity_data['user_friends'])
popularity_data.head()
```

    <ipython-input-19-3bc76a8cc0c0>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      popularity_data['followers_to_following'] = (popularity_data['user_followers'] / popularity_data['user_friends'])





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_description</th>
      <th>user_followers</th>
      <th>user_friends</th>
      <th>user_verified</th>
      <th>text</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>datetime</th>
      <th>VADER tweet</th>
      <th>VADER descr</th>
      <th>followers_to_following</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aggregator of Asian American news; scanning di...</td>
      <td>405</td>
      <td>1692</td>
      <td>False</td>
      <td>Same folks said daikon paste could treat a cyt...</td>
      <td>United States</td>
      <td>34.2192742</td>
      <td>-118.2318871</td>
      <td>2020-12-20 06:06:44</td>
      <td>0.4019</td>
      <td>0.0000</td>
      <td>0.239362</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Marketing dude, tech geek, heavy metal &amp; '80s ...</td>
      <td>834</td>
      <td>666</td>
      <td>False</td>
      <td>While the world has been on the wrong side of ...</td>
      <td>United States</td>
      <td>37.7790262</td>
      <td>-122.419906</td>
      <td>2020-12-13 16:27:13</td>
      <td>-0.1027</td>
      <td>0.3182</td>
      <td>1.252252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heil, hydra 🖐☺</td>
      <td>10</td>
      <td>88</td>
      <td>False</td>
      <td>#coronavirus #SputnikV #AstraZeneca #PfizerBio...</td>
      <td>España</td>
      <td>42.6851664</td>
      <td>-2.942752</td>
      <td>2020-12-12 20:33:45</td>
      <td>0.2500</td>
      <td>0.0000</td>
      <td>0.113636</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hosting "CharlesAdlerTonight" Global News Radi...</td>
      <td>49165</td>
      <td>3933</td>
      <td>True</td>
      <td>Facts are immutable, Senator, even when you're...</td>
      <td>Canada</td>
      <td>49.2608724</td>
      <td>-123.1139529</td>
      <td>2020-12-12 20:23:59</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>12.500636</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gastroenterology trainee, Clinical Research Fe...</td>
      <td>105</td>
      <td>108</td>
      <td>False</td>
      <td>Does anyone have any useful advice/guidance fo...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2020-12-12 20:11:42</td>
      <td>0.7003</td>
      <td>0.0000</td>
      <td>0.972222</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Let's graph this data

sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
verified = sns.scatterplot(x="VADER tweet", y="followers_to_following", data=popularity_data, palette="PuBuGn_d")

plt.show()
```


    
![output_34_0]({{"assets/vaccine-post/output_34_0.png" | relative_url }} " ")




```python
#It seems very scattered. Let's see the descriptive statistics.

popularity_data['followers_to_following'].describe()
```




    count      7326.000000
    mean        700.011158
    std        9412.303849
    min           0.000000
    25%           0.396727
    50%           0.873588
    75%           2.463399
    max      202322.000000
    Name: followers_to_following, dtype: float64



## The data seems to be very scattered, seemingly no correlation. This is caused by outliers such as celebrities in the data. Let's select only smaller set of data to make inferences.


```python
sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
verified = sns.scatterplot(x="VADER tweet", y="followers_to_following", data=popularity_data.loc[popularity_data['followers_to_following'] < 2.46]

, palette="PuBuGn_d")

plt.show()

print('### In the upper right quarter there seems to be higher density of data points. Let us select even smaller subset.')

subset = popularity_data.loc[popularity_data['followers_to_following'] < 2.46]
subset = subset.loc[subset['followers_to_following'] > 1]

sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
verified = sns.scatterplot(x="VADER tweet", y="followers_to_following", data=subset, palette="PuBuGn_d")

plt.show()
```


    

![output_37_0]({{"assets/vaccine-post/output_37_0.png" | relative_url }} " ")

    


    ### In the upper right quarter there seems to be higher density of data points. Let us select even smaller subset.



    

![output_37_2]({{"assets/vaccine-post/output_37_2.png" | relative_url }} " ")

    


## The scatter plot does not seem to have any correlation between data points. Perhaps there is a slight tencendy for the people with followers-to-following ratio between 1.6 to 2.5 to be positive about the vaccine. Let's run a regression analysis to test this.

## Regression analysis


```python
##inspired by https://www.statology.org/simple-linear-regression-in-python/
subset = subset.loc[subset['followers_to_following'] > 1.6]

import statsmodels.api as sm

#define response variable
y = subset['followers_to_following']

#define explanatory variable
x = subset['VADER tweet']

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

#define figure size
fig = plt.figure(figsize=(12,8))

#produce residual plots
fig = sm.graphics.plot_regress_exog(model, 'VADER tweet', fig=fig)

plt.show()
```

                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     followers_to_following   R-squared:                       0.000
    Model:                                OLS   Adj. R-squared:                 -0.002
    Method:                     Least Squares   F-statistic:                    0.1748
    Date:                    Sat, 07 Aug 2021   Prob (F-statistic):              0.676
    Time:                            17:54:43   Log-Likelihood:                -5.4052
    No. Observations:                     455   AIC:                             14.81
    Df Residuals:                         453   BIC:                             23.05
    Df Model:                               1                                         
    Covariance Type:                nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const           1.9614      0.012    166.311      0.000       1.938       1.985
    VADER tweet     0.0134      0.032      0.418      0.676      -0.049       0.076
    ==============================================================================
    Omnibus:                       93.315   Durbin-Watson:                   1.968
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               35.293
    Skew:                           0.481   Prob(JB):                     2.17e-08
    Kurtosis:                       2.032   Cond. No.                         2.80
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    

![output_39_1]({{"assets/vaccine-post/output_39_1.png" | relative_url }} " ")



### As suspected from the plot, the R-squared is close to zero with very small coefficient.

### There is no correlation between the followers-to-following ratios and the sentiment to vaccine!

## Plotting the sentiment across time.


```python
tweet_data['datetime'].describe()
```

    <ipython-input-24-490c9fec9c96>:1: FutureWarning: Treating datetime data as categorical rather than numeric in `.describe` is deprecated and will be removed in a future version of pandas. Specify `datetime_is_numeric=True` to silence this warning and adopt the future behavior now.
      tweet_data['datetime'].describe()





    count                    7353
    unique                   7348
    top       2021-01-11 15:40:25
    freq                        2
    first     2020-12-12 11:55:28
    last      2021-08-05 04:00:49
    Name: datetime, dtype: object




```python
#Plotting the raw data

sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
ax = sns.lineplot(x="datetime", y="VADER tweet", data=tweet_data, palette="PuBuGn_d")

plt.show()

```


    
![output_43_0]({{"assets/vaccine-post/output_43_0.png" | relative_url }} " ")




```python
#Moving average: All data

sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
ax = sns.lineplot(x="datetime", y="VADER tweet", data=tweet_data[['datetime','VADER tweet']].groupby(pd.Grouper(key='datetime',freq='M')).mean(), palette="PuBuGn_d")
ax2 = sns.lineplot(x="datetime", y="VADER tweet", data=tweet_data[['datetime','VADER tweet']].groupby(pd.Grouper(key='datetime',freq='W')).mean(), palette="PuBuGn_d")
plt.legend(labels=["Monthly moving average","Weekly moving average"])

ax.set_title('Vader-sentiment of Pfizer Vaccine tweets during 2021 (All users)')
ax.set_xlabel('Date')
ax.set_ylabel('Vader-score')

plt.show()

#Moving average: among users with following 50-1000 users and 50-1000 followers

subset2 = tweet_data.loc[tweet_data['user_followers'] < 1000]
subset2 = subset2.loc[subset2['user_followers'] > 50]
subset2 = subset2.loc[subset2['user_friends'] > 50]
subset2 = subset2.loc[subset2['user_friends'] < 1000]

sns.set(style="darkgrid")
sns.set(font_scale=0.85)
            
ax = sns.lineplot(x="datetime", y="VADER tweet", data=tweet_data[['datetime','VADER tweet']].groupby(pd.Grouper(key='datetime',freq='M')).mean(), palette="PuBuGn_d")
ax2 = sns.lineplot(x="datetime", y="VADER tweet", data=tweet_data[['datetime','VADER tweet']].groupby(pd.Grouper(key='datetime',freq='W')).mean(), palette="PuBuGn_d")
plt.legend(labels=["Monthly moving average","Weekly moving average"])

ax.set_title('Vader-sentiment of Pfizer Vaccine tweets during 2021 (Users with 50-1000 followers and following)')
ax.set_xlabel('Date')
ax.set_ylabel('Vader-score')

plt.show()
```


    
![output_44_0]({{"assets/vaccine-post/output_44_0.png" | relative_url }} " ")
    



    
![output_44_1]({{"assets/vaccine-post/output_44_1.png" | relative_url }} " ")
    


### The results are very similar. There's no reason to suspect that the news accounts affect the general sentiment in this dataset. The sentiment is quite stable, but the moving average is at all-time low on August 2021.

## Text cleanup using PorterStemmer and plotting the common words.

These lists are not 100% correct, because words were arbitarily removed. These removed words included for example, common words that provide no useful information for our analysis as well as the vaccine company names.


```python
cleaned_text = pd.DataFrame(columns=['sentence','vader_score'])
cleaned_text['vader_score']  = tweet_data['VADER tweet']
cleaned_text['sentence']  = tweet_data['text']

dataframe = cleaned_text
how_many_words = 25
tweet_id_in_df = 'vader_score'


class CleanText(BaseEstimator, TransformerMixin):
           
#Original cleanup script inspired by my university's NLP tutorial lesson, heavily edited.

    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    #uninteresting words, found via trial and error
    def remove_common_words(self, input_text):
        common_list = ['covid','pfizer','pfizerbiontech','vaccin','covidvaccin','biontech','coronaviru','http','co','pfizervaccin','astrazeneca']
        whitelist2 = []
        words = input_text.split() 
        clean_words = [word for word in words if (word not in common_list or word in whitelist2) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming).apply(self.remove_common_words)
        return clean_X
    
ct = CleanText()

#Good tweets, all people

sent_clean = ct.fit_transform(dataframe.loc[dataframe[tweet_id_in_df] > 0].sentence)
empty_clean = sent_clean == ''
sent_clean.loc[empty_clean] = '[no_text]'

cv = CountVectorizer()
bow = cv.fit_transform(sent_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(how_many_words), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
bar_freq_word = sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
ax.set_title('Common words tweeted by everyone in dataset, Vader-score > 0')
plt.show();

#Bad tweets, all people

#change condition
sent_clean = ct.fit_transform(dataframe.loc[dataframe[tweet_id_in_df] < 0].sentence)
empty_clean = sent_clean == ''
sent_clean.loc[empty_clean] = '[no_text]'

cv = CountVectorizer()
bow = cv.fit_transform(sent_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(how_many_words), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
bar_freq_word = sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
ax.set_title('Common words tweeted by everyone in dataset, Vader-score < 0')
plt.show();
```


    
![output_48_0]({{"assets/vaccine-post/output_48_0.png" | relative_url }} " ")
    



    
![output_48_1]({{"assets/vaccine-post/output_48_1.png" | relative_url }} " ")
    


#### There doesn't seem to be big difference between the two groups. The results are very predictable and interesting.

- For the tweets with positive sentiment, most common words included feelings of thankfulness and receiving their dose. Words such as "protect" and "like" were used.

- For the tweets with negative sentiment, the most common words were negatives such as no and not. There seems to be many tweets about common side effects such as "arm", "sore" and "side", referring to soreness in arm after the shot.

- The words with negative sentiment have significantly less frequency, as the sentiment is generally positive.






```python
#Good tweets, normal people
print('Good tweets, normal people')

#adjust
dataframe = subset2
how_many_words = 25
tweet_id_in_df = 'VADER tweet'

#change condition
sent_clean = ct.fit_transform(dataframe.loc[dataframe[tweet_id_in_df] > 0].text)
empty_clean = sent_clean == ''
sent_clean.loc[empty_clean] = '[no_text]'

cv = CountVectorizer()
bow = cv.fit_transform(sent_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(how_many_words), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
bar_freq_word = sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
ax.set_title('Common words tweeted by users with 50-1000 followers/following, Vader score > 0')
plt.show();

#Bad tweets, normal people
print('Bad tweets, normal people')

#change condition
sent_clean = ct.fit_transform(dataframe.loc[dataframe[tweet_id_in_df] < 0].text)
empty_clean = sent_clean == ''
sent_clean.loc[empty_clean] = '[no_text]'

cv = CountVectorizer()
bow = cv.fit_transform(sent_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(how_many_words), columns = ['word', 'freq'])

fig, ax = plt.subplots(figsize=(12, 10))
bar_freq_word = sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
ax.set_title('Common words tweeted by users with 50-1000 followers/following, Vader score < 0')
plt.show();
```

    Good tweets, normal people



    
![output_50_1]({{"assets/vaccine-post/output_50_1.png" | relative_url }} " ")
    


    Bad tweets, normal people



    
![output_50_3]({{"assets/vaccine-post/output_50_3.png" | relative_url }} " ")
    


## Conclusions

- Users in Canada were tweeting the most positive tweets about the vaccine, followed by UK, US, Malaysia and India.
- Verified users were surprisingly tweeting more negative tweets about the vaccine, this is however easily explained with the fact that many of these were news accounts.
- The followers to following ratio does not seem to play any role with the sentiment.
- The sentiment across time is quite stable and was at its highest around March, and is currently at it's lowest in August of 2021.
- The most common words for the positive tweets included feelings of thankfulness and words such as "protect".
- The most common words for the negative tweets included neutral negative words, as well as words associated with common side effects such as sore arm.





