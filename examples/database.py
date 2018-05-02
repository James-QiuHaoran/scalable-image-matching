from os import listdir
from os.path import isfile, join

# path_to_directory = "/home/jamesqiu/Desktop/test_in_lab"
path_to_directory = "/home/jamesqiu/Desktop/test-set2"

images = [f for f in listdir(path_to_directory) if isfile(join(path_to_directory, f)) and f[0] != '.']

from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES

es = Elasticsearch()
ses = SignatureES(es)

print("Adding images to the database...")
count = 0
for image in images:
    count += 1
    print(image + " - [ " + str(count) + " / " + str(len(images)) + " ]")
    ses.add_image(path_to_directory + "/" + image)
print("Done")
