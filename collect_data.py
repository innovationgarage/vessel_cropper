from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import config
import requests
import re
import os

def findShipES(imo_no):
    client = Elasticsearch(
        [
            "http://{}:{}@{}".format(config.user, config.pswd, config.host)
        ],
        verify_certs=True
    )    
    s = Search(using=client, index=config.index) \
        .filter("term", imo=imo_no)
    try:
        response = s.execute()
    except:
        return None
    if response.hits:
        try:
            gallery = response[0].gallery
            urls = [x.file for x in gallery]
        except:
            return None
        if len(urls) > 20:
            print("I found {} images. This could take a while!".format(len(urls)))
        else:
            print("I found {} images. It's gonna be fast!".format(len(urls)))
        return urls
    else:
        return None

def findShipSS(imo_no):
    url = "http://www.shipspotting.com/gallery/search.php?search_imo={}".format(imo_no) 
    page = requests.get(url)
    text = ''.join(page.text)
    exp = '<img src="http://www.shipspotting.com/photos/small'
    starts = [m.start() for m in re.finditer(exp, text)]
    length = 70
    substrings = [text[start:start+length] for start in starts]
    small_urls = [re.findall(r'"(.*?)"', string)[0] for string in substrings]
    middle_urls = [el.replace("small", "middle") for el in small_urls]
    return small_urls, middle_urls
    
                    
