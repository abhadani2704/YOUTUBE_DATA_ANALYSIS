#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing relevent and most used packages 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


# extracting and cleaning the comments file 

comments = pd.read_csv(r'/Users/abhishekbhadani/Desktop/UScomments.csv',error_bad_lines=False)


# In[4]:


comments.isnull().sum()


# In[5]:


comments.dropna(inplace=True)


# In[6]:


comments.isnull().sum()


# In[7]:


# installing the textblob package for seentiment analysis 

pip install textblob


# In[8]:


# performing sentiment analysis using textblob package as it assigns a value ranging from -1.0 to 1.0 after analyzing the sentiments of the text

from textblob import TextBlob


# In[9]:


comments.head(7)


# In[10]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️	").sentiment.polarity


# In[11]:


polarity = []
for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[12]:


print(len(polarity))


# In[13]:


sample_df = comments[:1000]


# In[14]:


comments['polarity'] = polarity 


# In[15]:


filter1 = comments['polarity'] == 1.0
filter2 = comments['polarity'] == -1


# In[16]:


sample_df.head()


# In[17]:


positive = comments[filter1] 


# In[18]:


negative = comments[filter2]


# In[40]:


# installing and using wordcloud package ,it removes the irrelevent text form the string like 'is','the' etc

pip install wordcloud


# In[19]:


from wordcloud import WordCloud,STOPWORDS


# In[20]:


WordCloud()


# In[21]:


total_comments_positive = " ".join(positive['comment_text'])


# In[22]:


wordcloud = WordCloud(stopwords = set(STOPWORDS)).generate(total_comments_positive)


# In[23]:


# expressing parts of comments containing positive words 

plt.imshow(wordcloud)
plt.axis('off')


# In[24]:


total_comments_negative = " ".join(negative['comment_text'])


# In[25]:


wordcloud2 = WordCloud(stopwords = set(STOPWORDS)).generate(total_comments_negative)


# In[26]:


# expressing parts of comments containing negative words 

plt.imshow(wordcloud2)
plt.axis('off')


# In[64]:


# installing emoji package for emoji analysis

pip install emoji==2.2.0


# In[27]:


import emoji
emoji.__version__


# In[28]:


comments["comment_text"].head()


# In[29]:


comment = 'trending 😉'
all_emoji_list = [char for comment in comments['comment_text'].dropna() for char in comment if char in emoji.EMOJI_DATA]
    


# In[30]:


all_emoji_list[0:10]


# In[31]:


from collections import Counter


# In[32]:


common_emojis = Counter(all_emoji_list).most_common(10)


# In[33]:


emojis = [common_emojis[i][0] for i in range(10)]
emojis    


# In[34]:


frequency = [common_emojis[i][1] for i in range(10)]
frequency   


# In[35]:


import plotly.graph_objs as go 
from plotly.offline import iplot


# In[39]:


trace = go.Bar(x = emojis,y = frequency )
trace


# In[40]:


# expressing most used emoji in the comments as a bar graph

iplot([trace])


# In[42]:


# accessing all files in the particular directory saved in your pc 

import os 


# In[45]:


files = os.listdir('/Users/abhishekbhadani/Desktop/additional_data')
files


# In[47]:


# extracting all the csv files saved in that file 

files_csv = [file for file in files if '.csv' in file]


# In[48]:



import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[101]:


# merging all csv in one dataframe object

full_df = pd.DataFrame()


# In[102]:


path = '/Users/abhishekbhadani/Desktop/additional_data'


# In[103]:


for file in files_csv:
    
    current_df = pd.read_csv(path+'/'+file,encoding='iso-8859-1',error_bad_lines=False)
    full_df = pd.concat([full_df,current_df],ignore_index = True)


# In[105]:


full_df.shape


# In[58]:


full_df.head()


# In[106]:


full_df[full_df.duplicated()].shape


# In[107]:


full_df = full_df.drop_duplicates()


# In[108]:


full_df.shape


# In[68]:


# loading the dataframe into csv file in a particular directory in your pc

full_df[0:1000].to_csv('/Users/abhishekbhadani/Desktop/youtube_data/youtube_sample.csv',index = False)


# In[69]:


# loading the dataframe into json file in a particular directory in your pc

full_df[0:1000].to_json('/Users/abhishekbhadani/Desktop/youtube_data/youtube_sample.json')


# In[71]:


# creating a engine to load the dataframe into a sql database

from sqlalchemy import create_engine


# In[73]:


engine = create_engine(r'sqlite:////Users/abhishekbhadani/Desktop/youtube_data/youtube_sample.sqlite')


# In[75]:


# adding a table named user into the table created by the engine

full_df[0:1000].to_sql('users',con=engine,if_exists='append')


# In[77]:


full_df['category_id'].unique()


# In[82]:


# this is to analyze the views ,likes, dislikes of each category of videoes on youtube  

json_df = pd.read_json('/Users/abhishekbhadani/Desktop/additional_data/US_category_id.json')
json_df['items'][0]


# In[88]:


# extracting the category of the vedioes based on their id 
cat_dict = {}
for item in json_df['items'].values:
    cat_dict[int(item['id'])] = item['snippet']['title']


# In[89]:


cat_dict


# In[109]:


# adding the category according to the id in dataframe 
full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[110]:


full_df.head()


# In[119]:


# creating a function which will create a boxplot usung full_df dataframe and required axes and desirable size
def express_box(x = 'category_name',y = 'likes',fig_size = (14,9)):
    plt.figure(figsize=fig_size)
    sns.boxplot(x=x , y = y , data=full_df)
    plt.xticks(rotation = 'vertical')
#  expressing a boxplot of category_name vs likes  
express_box()


# In[120]:


# adding like_rate , dislike_rate and comment_rate wrt views to existing dataframe 
full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[121]:


# like_rate vs category box_plot
express_box(y = 'like_rate',fig_size = (15,7))


# In[122]:


# showing how the views of a particular video increases based upon the likes

sns.regplot(x = 'views' , y = 'likes' , data = full_df)


# In[123]:


sns.regplot(x = 'views',y = 'dislikes',data = full_df)


# In[129]:


corr_table = full_df[['views','likes','dislikes']].corr()


# In[130]:


sns.heatmap(corr_table,annot = True)


# In[138]:


full_df['channel_title'].value_counts()


# In[143]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()
cdf = cdf.rename(columns = {0:'total_videos'})


# In[144]:


import plotly.express as px


# In[145]:


# total views of each channel 
px.bar(data_frame = cdf[0:20],x = 'channel_title',y = 'total_videos')
px.bar()


# In[146]:


# analyzing how the views are affected based on the punctuation marks contained in the vedio title

import string


# In[147]:


string.punctuation


# In[149]:


[char for char in full_df['title'][0] if char in string.punctuation]


# In[150]:


full_df.head(10)


# In[152]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[155]:


full_df['punc_count'] = full_df['title'].apply(punc_count)


# In[156]:


full_df


# In[157]:


express_box(x = 'punc_count',y = 'views',fig_size = (8,6))

