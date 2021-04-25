#%%
import facebook
import pandas as pd
import re, time, requests
from selenium import webdriver
import sys
from bs4 import BeautifulSoup
#%%
def FindLinks(url, n):
    Links = []
    driver.get(url)
    for i in range(n):
        time.sleep(1)
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    #driver.find_element_by_xpath('//a[@id="expanding_cta_close_button"]').click()
    soup = BeautifulSoup(driver.page_source)
    #posts = soup.findAll('div', {'class':'clearfix y_c3pyo2ta3'})
    posts = soup.findAll('div', {'class':'clearfix _42ef'})
    for i in posts:
        Links.append('https://www.facebook.com' + i.find('a',{'class':'_5pcq'}).attrs['href'].split('?',2)[0])
    return Links,posts
#%%
#需安裝 Chrome WebDriver
driver = webdriver.Chrome()
driver.get('https://zh-tw.facebook.com/')
#%%
#已爬粉絲團紀錄 資料筆數
#Links = FindLinks(url = 'https://www.facebook.com/groups/556944001110781/',n = 1000)
#Links = FindLinks(url = 'https://www.facebook.com/groups/688953464636855/',n = 1000)
#Links = FindLinks(url = 'https://www.facebook.com/groups/2562678580439603/',n = 1000)
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/goodwaytomakemoney/',n = 1000) 4624
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/1589367754651796/',n = 1000)   2225
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/688953464636855/',n = 1000)    1753
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/e07110711/',n = 1000)          4794
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/1043101689062715/',n = 1000)   4893
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/557196687737178/',n = 1000)    2851
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/1076475292384362/',n = 1000)   3280
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/598334243607534/',n = 2)
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/366124343804226//',n = 1000) 4534
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/wonderfulmakemoney/?ref=br_rs',n = 1000)
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/256239898478454/',n = 1000) 2074
#Links,posts = FindLinks(url = 'https://www.facebook.com/groups/432367753627995/',n = 1000) 5674
try:
    Links,posts = FindLinks(url = 'https://www.facebook.com/groups/1931313210453873/',n = 1000)
except:
    print("例外結束")
#%%
# print(posts)
# for i in posts:
#     print(i.find('a',{'class':'_5pcq'}).attrs['href'].split('?',2)[0])
#%%
#去除字串
def find_ch_en_num(file):
    pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    chinese = re.sub(pattern, '', file)
    return chinese
def expand(url):
    time.sleep(1)
    driver.get(url)
    try:
        driver.find_element_by_xpath('//a[@lang="en_US"]').click()
    except:
        print("Now is in EN_US")
    time.sleep(1)
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
a=[]
# 文章內容與互動摘要
def PostContent(soup):
    # po文區塊
    time.sleep(1)
    userContent = soup.find('div', {'class':'_5pcr userContentWrapper'})
    # po文人資訊區塊
    PosterInfo = userContent.find('div', {'class':'l_c3pyo2v0u i_c3pynyi2f clearfix'})
    time.sleep(1)
    # 文章內容
    try:
        Content = userContent.find('div', {'class':'_5pbx userContent _3576'}).text
        time.sleep(1)
        Content1=find_ch_en_num(Content)
    except:
        Content = ""
    if len(Content)>10:
        return pd.DataFrame(data = [{'Content':Content1}],columns = [ 'Content'])
        a.append(Content1)
#     print(Content1)

#%%
#%%
Links_0=Links
#備份貼文網址
f = open('file_links18.txt', 'w')
for i in range(0,len(Links)-1):
    f.write(Links[i]+"\n")
f.close()
result=[]
g=[]

with open('file_links18.txt','r') as f:
    line=(f.readlines())
    
try0=[line[0], line[1], line[2]]

for i in range(0,len(line)):
    g.append(line[i])
#%%
# proxy = webdriver.Proxy()
# proxy.http_proxy = '91.201.240.242'
# 所有貼文內容
#input()
PostsInformation = pd.DataFrame()
PostsComments = pd.DataFrame()
num=0
for i in g:
    num+=1
    if num%5==0:
        time.sleep(1)
    i.rstrip('\n')
    print('Dealing with: ' + i)
    try:
        time.sleep(0.5)
        expand(i)
        
        soup = BeautifulSoup(driver.page_source)
        
        #PostsComments = pd.concat([PostsComments, CrawlComment(soup)],ignore_index=True)
        try:
            PostsInformation = pd.concat([PostsInformation, PostContent(soup)],ignore_index=True)
        except:
            time.sleep((num%5)+1)
            #input()
            print("無內文")
    except:
        print('Load Failed: ' + i)

PostsInformation




PostsInformation.to_excel('./PostsInformation24.xlsx')

print("finish")
