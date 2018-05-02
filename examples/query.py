from elasticsearch import Elasticsearch
 
es = Elasticsearch()
res = es.search(index="images", doc_type="image", body={"query": {"match": {"batch": "18"}}})
print("%d documents found" % res['hits']['total'])
for doc in res['hits']['hits']:
    print("%s - %s" % (doc['_source']['path'], doc['_source']['batch']))
