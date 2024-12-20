# %%
import pandas as pd
import numpy as np


# %%


df = pd.read_csv(r'C:\Users\USER\Desktop\coding\Machine Learning\Projects\Email_Spam_Detection\spam.csv', encoding='latin1')
df

# %%
#Data Cleaning
#EDA
#Text Preprocessing
#Model Building
#Evaluation
#Imporvement
#Website(Streamlit)

# %%
#Data Cleaning
df.info()

# %%
#dropping last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# %%
df.head(5)

# %%
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

# %%
df.info()

# %%
#small note: Label Encoder changes values in the column itself hence we have used that while getdummies does the same thing by splitting it into different columns hence we didnt use it here
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df

# %%
#missing values
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
#remove duplicates
df=df.drop_duplicates(keep='first')
df.duplicated().sum()

# %%
df.shape

# %%
#EDA
df['target'].value_counts()

# %%
#Data is Imbalanced
import matplotlib.pyplot as plt
import nltk 
nltk.download('punkt')


# %%
df['num_characters']=df['text'].apply(len)

# %%
#num of words by using nltk(tokenize)
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df

# %%
#num of sentences nltk(tokenize)
df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df

# %%
df[['num_characters','num_words','num_sentences']].describe()

# %%
#checking the same for ham and spam messages seperately
#ham
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()

# %%
#spam
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()

# %%
#mean of spam messages are greater than mean of ham messages
import seaborn as sns

# %%
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')

#num of characters in spam are greater than of that of ham

# %%
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')

# %%
sns.histplot(df[df['target']==0]['num_sentences'])
sns.histplot(df[df['target']==1]['num_sentences'],color='red')

# %%
sns.pairplot(df,hue='target')

# %%
correlation_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(correlation_matrix, annot=True)


# %%
#Data Preprocessing
#lower case
#tokenize
#removing special characters
#removing special characters
#removing stop words and punctuation
#stemming

# %%
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear() #only including stopwords and without punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    #stemming it
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

# %%
transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')

# %%
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
import string
string.punctuation
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

# %%
df['text'][0]

# %%
df['transformed_text']=df['text'].apply(transform_text)
df.head()

# %%
#visulaizing all the important words
!pip install wordcloud


# %%
 #top 30 words for ham and spam
df.head()

# %%
spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)

# %%
from collections import Counter
top_30_spam = Counter(spam_corpus).most_common(30)
top_30_spam_df = pd.DataFrame(top_30_spam, columns=['Word', 'Count'])
plt.figure(figsize=(12, 6))
sns.barplot(x='Word', y='Count', data=top_30_spam_df)
plt.xticks(rotation='vertical')
plt.title('Top 30 Words in Spam Messages')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# %%
ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)

# %%
ham_corpus

# %%
from collections import Counter
top_30_ham = Counter(ham_corpus).most_common(30)
top_30_ham_df = pd.DataFrame(top_30_ham, columns=['Word', 'Count'])
plt.figure(figsize=(12, 6))
sns.barplot(x='Word', y='Count', data=top_30_ham_df)
plt.xticks(rotation='vertical')
plt.title('Top 30 Words in Spam Messages')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# %%
