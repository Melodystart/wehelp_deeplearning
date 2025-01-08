from urllib.request import urlopen
import json
import csv
import statistics

task1 = open('products.txt', 'w')
task2 = open('best-products.txt', 'w')

i5s = []
prices = []
ids = []

task4 = open('standardization.csv', 'w', newline='')
writer = csv.writer(task4)

response = urlopen('https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31&attr=&pageCount=40&page=1')
data = json.load(response)   
page = data['TotalPage']

for p in range(1, page+1):
  url = 'https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31&attr=&pageCount=40&page='+str(p)
  response = urlopen(url)
  data = json.load(response)

  Prods = data['Prods']
  for prod in Prods:
    if 'ASUS' in prod['Name'].upper():
      task1.write(prod['Id']+'\n')
      ids.append(prod['Id'])
      prices.append(prod['Price'])

      if prod['reviewCount'] != None and prod['ratingValue'] != None:
        if prod['reviewCount'] > 0 and prod['ratingValue'] > 4.9:
          task2.write(prod['Id']+'\n')

      if 'i5處理器' in prod['Name'].lower() or 'i5' in prod['Describe'].lower():
        i5s.append(prod['Price'])

task1.close()
task2.close()

i5Total = 0
for i5 in i5s:
  i5Total += i5
i5Average = i5Total / len(i5s)

# task3: average price of ASUS PCs with Intel i5 processor = 22,726.4
print(i5Average) 


mean = statistics.mean(prices)
pstdev = statistics.pstdev(prices)

writer.writerow(['ProductID','Price','PriceZScore'])
for i in range(len(ids)):
  writer.writerow([ids[i],prices[i],(prices[i]-mean)/pstdev])

task4.close()