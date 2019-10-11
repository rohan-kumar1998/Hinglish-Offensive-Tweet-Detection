import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
import re 
from nltk.tokenize import WordPunctTokenizer

df = pd.read_csv( 'HOT_Dataset_modified.csv', index_col=None, header=None, engine='python' )

cols = df.columns.tolist()

df = pd.DataFrame(df, columns = [0,1])
df.dropna(inplace= True)
df = df.rename(index=str, columns={0: 'score', 1 : 'text'})
df.to_csv('HOT_dataset.csv', index = False)

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text,'html.parser')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

clean = df.text

cleaned_data = []
for t in clean:
    cleaned_data.append(tweet_cleaner(t))

cleaned_data = pd.DataFrame(cleaned_data, columns = ['text'])
df = pd.read_csv( 'HOT_dataset.csv', index_col=None, header=0, engine='python' )
cleaned_data['score'] = df['score']

cleaned_data.to_csv('HOT_cleaned.csv', index = False)