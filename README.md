# scalable-image-matching

This is a image matching system(library) for scalable and efficient matching of images from a large database. The basic idea is to compute perceptural hash value for each image and compare the similarity based on the pHash computed. Searching are scalable with the elasticsearch as the backend database.

Online search and local search are both supported, i.e. provide URL or local address of the target image to the library.

## Get Started

- Clone this repo and enter the directory;
- Install with pip: `pip install numpy scipy .`;
- Install `Elasticsearch`, https://www.elastic.co/downloads/elasticsearch;
- Make sure elasticsearch is running before scalable searching is used;
