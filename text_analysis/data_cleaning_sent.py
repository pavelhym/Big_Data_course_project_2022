path = 'GME_with_comments_groupped_text'


from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import  SparkContext
from pyspark.sql.types import StructType

import pyspark.sql.types as Ts
from pyspark.sql import functions as F 


conf = SparkConf().setAppName('appName').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)


data = spark.read.option("delimiter", ",")\
                   .option("header", "true")\
                   .option("multiline", "true")\
                   .option("escape", "\\")\
                   .option("escape", '"')\
                   .option("quote", '"')\
                   .csv(path)
data = data.limit(1)
data.show()


'''Cleaning'''

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('words')
nltk.download('stopwords')
nltk.download('vader_lexicon')


stemmer = nltk.PorterStemmer()
lemm = nltk.WordNetLemmatizer()
# eng_words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))


def cleaning(x):
  x = re.sub(r"\/r|\/n|\/t", '', x)
  x = re.sub(r"\\n|\\t|\\r", '', x)
  x = re.sub('[^a-zA-Z]',' ', x)
  x = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", ' ', x)
  x = str(x).replace("\n", '')
  x = str(x).replace("/r", '')
  x = str(x).replace("[removed]", '')
  
  x=x.lower().split()
  # tweet = [w for w in tweet if w in eng_words or not w.isalpha()]
  x=[stemmer.stem(word) for word in x if (word not in stop_words)]
  # tweet=[lemm.lemmatize(word) for word in tweet if (word not in stop_words)]

  return ' '.join(x)

cleaning_udf = F.udf( lambda x: cleaning(x), returnType=Ts.StringType() )


data2 = data.withColumn( 'proc_comments', cleaning_udf('comments') )
data2.show()


'''Sentiments extraction'''

import pysentiment2 as ps2
import datetime

lm = ps2.LM()
hiv4 = ps2.HIV4()


lm_schema = Ts.MapType(
    keyType=Ts.StringType(), valueType=Ts.FloatType()
)

def lm_scoring(x):
  token_lm = lm.tokenize(x)
  score_lm = lm.get_score(token_lm)
  score_lm_2 = {
      'Positive': float(score_lm['Positive']),
      'Negative': float(score_lm['Negative']),
      'Polarity': float(score_lm['Polarity']),
      'Subjectivity': float(score_lm['Subjectivity'])
  }
  return score_lm_2

lm_scor_udf = F.udf( lambda x: lm_scoring(x), lm_schema )





def hiv4_scoring(x):
  token_h4 = lm.tokenize(x)
  score_h4 = lm.get_score(token_h4)
  score_h4_2 = {
      'Positive': float(score_h4['Positive']),
      'Negative': float(score_h4['Negative']),
      'Polarity': float(score_h4['Polarity']),
      'Subjectivity': float(score_h4['Subjectivity'])
  }
  return score_h4_2

h4_scor_udf = F.udf( lambda x: hiv4_scoring(x), returnType=Ts.MapType(Ts.StringType(), valueType=Ts.FloatType() ) )


def vader_scoring(x):
  token_vader = lm.tokenize(x)
  score_vader = lm.get_score(token_vader)
  score_vader_2 = {
      'Positive': float(score_vader['Positive']),
      'Negative': float(score_vader['Negative']),
      'Polarity': float(score_vader['Polarity']),
      'Subjectivity': float(score_vader['Subjectivity'])
  }
  return score_vader_2
  
vader_scor_udf = F.udf( lambda x: vader_scoring(x), returnType=Ts.MapType(keyType=Ts.StringType(), valueType=Ts.DoubleType() ) )


data2 = data.withColumn( 'comments_lm_scoring_dict', lm_scor_udf(F.col('comments')) )\
            .withColumn( 'lm_scor_Positive', F.col('comments_lm_scoring_dict')['Positive'])\
            .withColumn( 'lm_scor_Negative', F.col('comments_lm_scoring_dict')['Negative'])\
            .withColumn( 'lm_scor_Polarity', F.col('comments_lm_scoring_dict')['Polarity'])\
            .withColumn( 'lm_scor_Subjectivity', F.col('comments_lm_scoring_dict')['Subjectivity'])\
            .drop('comments_lm_scoring_dict')
            
data2.show()
