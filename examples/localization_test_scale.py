import time
from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES

'''
def image_match(target, threshold=0.7):
    es = Elasticsearch()
    ses = SignatureES(es, distance_cutoff=threshold)

    print("Target: " + target)

    results = ses.search_image(target)
    print("Searching ... ")
    most = 1
    tartet = ""
    for result in results:
        if result['dist'] < most:
            most = result['dist']
            target = result['path']
        print(result['path'] + " - score: " + str(result['score']) + " - dist: " + str(result['dist']))
    print("\nThe most similar image is: " + target + " with distance = " + str(most))
    return target
'''

es = Elasticsearch()
ses = SignatureES(es, distance_cutoff=0.7)

target = "/home/jamesqiu/Desktop/first_try/3.JPG"
print("Target: " + target)

start_time = time.time()
results = ses.search_image(target)

print("Searching ... ")
most = 1
tartet = ""
for result in results:
    if result['dist'] < most:
        most = result['dist']
        target = result['path']
    print(result['path'] + " - score: " + str(result['score']) + " - dist: " + str(result['dist']))
print("\nThe most similar image is: " + target + " with distance = " + str(most))

print("--- %s seconds ---" % (time.time() - start_time))
