import pandas as pd
import glob
import os
import numpy as np
from astropy.table import vstack, Table


path = r'D:\\Documents\\ITMO\\Year1\\Big_Data_course_project_2022\\storage' # use your path
all_files = glob.glob(os.path.join(path , "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

all = pd.concat(li, axis=0, ignore_index=True)
all.columns
len(np.unique(all['Post ID']).tolist())
all['length'] = all['Comments'].str.len()

#Subs nulls
all['length'] = all['length'].fillna(0)


max_df = all.groupby(['Post ID'])['length'].agg('max').reset_index()
new_df = pd.merge(all, max_df,  how='inner', left_on=['Post ID','length'], right_on = ['Post ID','length'])
new_df = new_df.drop_duplicates(['Post ID'])
#comments concatenated
new_df.to_csv('comments_ready_1.csv')


posts = pd.read_csv('posts_first.csv')
comments = pd.read_csv('comments_ready_1.csv')

comments.columns
comments =  comments.drop('Unnamed: 0.1', 1).drop('Unnamed: 0', 1)
posts2 = pd.merge(posts, comments,  how='left', left_on=['Post ID'], right_on = ['Post ID'])
posts2["Publish Date"] =  pd.to_datetime(posts2["Publish Date"])
df.groupby(['name', 'month'], as_index = False).agg({'text': })
posts2 = posts2.replace(np.nan,'',regex=True)
posts2["Publish Date"] = posts2["Publish Date"].dt.date
posts3 = posts2.groupby(['Publish Date'], as_index = False).agg({'Score': 'sum','Total No. of Comments': 'sum', 'Title' : ' '.join, 'Comments' : ' '.join, 'selftext' : ' '.join })
type(posts2['Comments'][0])
posts3.columns = ['date', 'likes', 'comments_num', 'titles', 'comments','selftext']
prices = pd.read_csv('GME.csv')
prices["Date"] =  pd.to_datetime(prices["Date"])

#get df with all columns
final  = pd.merge(posts3, prices,  how='left', left_on=['date'], right_on = ['Date']).drop('Date',1)

final.to_csv('storage/GME_with_comments_groupped.csv')


#Final merge
posts = pd.read_csv('D://Documents//ITMO//Year1//Course_project_github//Big_Data_course_project_2022//storage//GME_with_comments_groupped.csv')

posts.columns.tolist()

needed_columns = [
 'date',
 'likes',
 'comments_num',
 'posts_num',
 'Adj Close',
 'Volume']

posts = posts[needed_columns]

text_processed = pd.read_csv('D://Documents//ITMO//Year1//Course_project_github//Big_Data_course_project_2022//storage//text_processed.csv')


needed_columns = [
 'date',
 'comments_lm_pos',
 'comments_lm_neg',
 'comments_lm_pol',
 'comments_lm_subj',
 'comments_hiv4_pos',
 'comments_hiv4_neg',
 'comments_hiv4_pol',
 'comments_hiv4_subj',
 'comments_vader_neg',
 'comments_vader_compound',
 'comments_vader_pos',
 'comments_vader_neu',
]

text_processed = text_processed[needed_columns]

emojies = pd.read_csv('D://Documents//ITMO//Year1//Course_project_github//Big_Data_course_project_2022//storage//emoji_dataset_new.csv').drop('Unnamed: 0', axis = 1)

posts["date"] =  pd.to_datetime(posts["date"])
text_processed['date'] = text_processed['date'].astype('datetime64')
emojies['date'] = emojies['date'].astype('datetime64')



only_values =  pd.merge(posts, text_processed,  how='inner', left_on=['date'], right_on = ['date'])

only_values = pd.merge(only_values, emojies,  how='left', left_on=['date'], right_on = ['date'])

only_values = only_values.fillna(0)

only_values.to_csv('D://Documents//ITMO//Year1//Course_project_github//Big_Data_course_project_2022//storage//GME_only_values.csv')