from psaw import PushshiftAPI
import datetime as dt
import xlsxwriter

#name the file what you want
workbook = xlsxwriter.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet()

api = PushshiftAPI()
#choose the date you want to start from yyyy-mm-dd
start_epoch=int(dt.datetime(2020, 4, 10).timestamp())

#use the .txt file with the phrases you want
words = open("finance_words.txt","r").read().splitlines()

col = 0
#create one column at a time for each search word
for text in words:
    col = col + 1
    print(text)
    #choose the subreddit you want to search
    gen = api.search_comments(after = start_epoch, q = text, subreddit='worldnews', aggs = 'created_utc', frequency = 'day', size = '0')

    cache = []

    for c in gen:
        cache.append(c)
    row = 0
    worksheet.write(row,col,text)
    for x in cache[0]['created_utc']:
        row = row + 1
        worksheet.write(row,col,x['doc_count'])

workbook.close()