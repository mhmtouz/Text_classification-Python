# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:18:54 2019

@author: black
"""
import urllib3
from bs4 import BeautifulSoup


text=open("data_.txt").read()
for article_url in text[:5]:
  title, text = get_only_text(article_url)
  print ('----------------------------------')
  print (title)
  for s in fs.summarize(text, 2):
   print ('*',s)