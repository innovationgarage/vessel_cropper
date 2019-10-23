from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import config

def findShip(imo_no):
    client = Elasticsearch(
        [
            "http://{}:{}@{}".format(config.user, config.pswd, config.host)
        ],
        verify_certs=True
    )    
    s = Search(using=client, index=config.index) \
        .filter("term", imo=imo_no)
    response = s.execute()
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
