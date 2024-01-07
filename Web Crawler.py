# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import csv
import re
from datetime import datetime

hotSearchNum = 50 #爬前5个热搜(最多50个)
pageNum = 20 #爬评论的页数
pageNumUser = 5
ifTimeIndex = 1 #是否按时间排序


#获得热搜搜索界面url

aheader = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0',
    'Cookie':'UOR=www.baidu.com,s.weibo.com,www.baidu.com; SINAGLOBAL=9415934088543.445.1691808757132; ULV=1692369298432:3:3:2:5101815058994.349.1692369298369:1692028644435; ALF=1723564671; SUB=_2A25J2KtdDeThGeNM71sZ-CvOzz-IHXVrIjUVrDV8PUJbkNANLVXBkW1NTgO_tD8E4GeuQq0369jfTCFRkhNX6R6g; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWrn-ag3vJqRQuE2y7R7mZR5NHD95QfeoB41hnfeoB0Ws4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNSKzX1KnRSKzXe7tt; _s_tentry=www.baidu.com; Apache=5101815058994.349.1692369298369'
    }
#注意！这里填登陆以后f12看到的UserAgent和Cookie

aText = requests.get("https://weibo.com/ajax/side/hotSearch",headers = aheader).text
rule = "#.{0,20}#"
arrayHotKey = re.findall(rule,aText)

#得到热搜搜索url列表arrayHotUrl   



#获得内容,清洗数据
def washData(_key,_n):
    #print(_url)
    if ifTimeIndex == 0:
        _url = "https://s.weibo.com/weibo?q="+_key[1:-1]+"&Refer=index"
    else:
        _url = "https://s.weibo.com/realtime?q="+_key[1:-1]+"&rd=realtime&tw=realtime&Refer=weibo_realtime"
    _hotText = requests.get(_url,headers = aheader).text    
    if _n >= 2:
        for i in range(2,_n+1):
            #print(_url+"&page="+str(i))
            _hotTextPage = requests.get(_url+"&page="+str(i),headers = aheader).text 
            _hotText += _hotTextPage 
            
    _dirtyHotText = re.findall("</a>.{0,500}</p>",str(_hotText))
    _hotText = [_key]
    for _singleText in _dirtyHotText:
        _singleText = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',_singleText)
        _singleText = ''.join(_singleText)
        _hotText.append(_singleText)

    #print(_hotText)


    return _hotText

def washDataUser(_key,_n):
    #print(_url)
    if ifTimeIndex == 0:
        _url = "https://s.weibo.com/weibo?q="+_key[1:-1]+"&Refer=index"
    else:
        _url = "https://s.weibo.com/realtime?q="+_key[1:-1]+"&rd=realtime&tw=realtime&Refer=weibo_realtime"
    _hotText = requests.get(_url,headers = aheader).text  
    #print(_url)
    #print(_hotText)
    if _n >= 2:
        for i in range(2,_n+1):
            #print(_url+"&page="+str(i))
            _hotTextPage = requests.get(_url+"&page="+str(i),headers = aheader).text 
            _hotText += _hotTextPage 
            
    pattern = r'a href="(.*?)"\s+nick-name='
    _dirtyHotText = re.findall(pattern,str(_hotText))
    

    #print(_dirtyHotText)

    pattern = r'"(.*?)"\s+class='
    User_url = ( re.findall(pattern," ".join(_dirtyHotText)))
    pattern = r'/(\d+)\?'
    User_url = ( re.findall(pattern," ".join(User_url)))  

    User_url_clean = []
    all_text = []
    i = 0
    for text in User_url:
        i+=1
        #print(i)
        user_url = "https://weibo.com/ajax/profile/info?custom="+ text
        #print(user_url)
        user_text = requests.get(user_url,headers = aheader).text  
        
        _singleText = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',user_text)
        _singleText = ''.join(_singleText)
        if("\"gender\":\"f\"" in user_text):
            _singleText = _singleText + "女"
        if("\"gender\":\"m\"" in user_text):
            _singleText = _singleText + "男"
        all_text.append(_singleText)
    all_text = "".join(all_text)

    #print(_hotText)
    with open("画像.csv", 'w', newline='') as hotCsv:
        writer = csv.writer(hotCsv)
        writer.writerows([[all_text]])

    return all_text


if ifTimeIndex == 0:
    rt = ''
else:
    rt = 'T'
        
with open(rt+str(datetime.now())[:10]+'.csv', 'w', newline='') as hotCsv:
    writer = csv.writer(hotCsv)
    for hotKey in arrayHotKey[:hotSearchNum]:
        hotText = washData(hotKey,pageNum)
        writer.writerows([hotText])
        print("finish")
    
with open(rt+str(datetime.now())[:10]+'User.csv', 'w', newline='') as hotCsv:
    writer = csv.writer(hotCsv)
    for hotKey in arrayHotKey[:hotSearchNum]:
        hotText = [hotKey,washDataUser(hotKey,pageNumUser)]
        writer.writerows([hotText])
        print("finish")
        

print("ok")