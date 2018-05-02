import time, sys
from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES

es = Elasticsearch()
ses = SignatureES(es, distance_cutoff=0.6)

target = sys.argv[1]
print("Target: " + target)

start_time = time.time()
results = ses.search_image(target)

print("Searching ... ")
most = 1
tartet = ""
for result in results:
    if result['dist'] < most:
        most = result['dist']
        path_most = result['path']
    print(result['path'] + " - score: " + str(result['score']) + " - dist: " + str(result['dist']))
# print("\nThe most similar image is: " + target + " with distance = " + str(most))
name = path_most[path_most.rfind('/')+1:]
x = name[:name.find('_')]
tmp = name[name.find('_')+1:]
y = tmp[:tmp.find('_')]
d = tmp[tmp.find('.')-1:tmp.find('.')]
batch = tmp[tmp.find('_')+1:tmp.rfind('_')]
print("\nImage: " + name + " - x: " + x + " y: " + y + " direction: " + d)
print("\n--- %s seconds ---\n" % (time.time() - start_time))

if d == "1" or d == "2" or d == "3":
    # return all signatures along the path ahead
    print("Signatures on the path ahead: path - batch - direction - signature\n")
    for i in range(int(batch)-1):
        res = es.search(index = "images", 
                        doc_type = "image", 
                        body = {
                            "query": {
                                "match": {
                                    "batch": str(i+1),
                                }
                            }
                        })
        # print("%d documents found" % res['hits']['total'])
        count = 0
        for doc in res['hits']['hits']:
            path = doc['_source']['path']
            d = path[path.find('.')-1:path.find('.')]
            if d in ['1', '2', '3']:
                print("[P]%s[/P] - [B]%s[/B] - [D]%s[/D]" % (path, doc['_source']['batch'], d))
                print("[S]" + str(doc['_source']['signature']) + "[/S]\n")
                count += 1
        # print(str(count) + " records found!\n")
