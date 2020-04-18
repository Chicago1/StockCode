from psaw import PushshiftAPI
from datetime import datetime
from pandas import DataFrame as df

api = PushshiftAPI()
#choose the date you want to start from yyyy-mm-dd
start_epoch=int(datetime(2020, 4, 10).timestamp())

#use the .txt file with the phrases you want
words = open("finance_words.txt","r").read().splitlines()

#DataFrame to write the CSV

reddit_data = {}
#create one column at a time for each search word
for text in words:
    print(text)
    #choose the subreddit you want to search
    gen = api.search_comments(after = start_epoch, q = text, subreddit='worldnews', aggs = 'created_utc', frequency = 'day', size = '0')
    
    for item_list in gen:        
        for item in item_list['created_utc']:
            date = datetime.utcfromtimestamp(item['key']).strftime('%Y-%m-%d')
            if reddit_data.get(date) is None:
                reddit_data[date] = {}
            reddit_data[date][text] = item['doc_count']

reddit_df = df.from_dict(reddit_data,orient='index')
reddit_df.to_csv("reddit.csv",index=True, header=True)   