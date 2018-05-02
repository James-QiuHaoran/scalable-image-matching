import sys
from image_match.goldberg import ImageSignature
import numpy

gis = ImageSignature()

number = int(sys.argv[1])
target = sys.argv[2]
print("Target:" + target)
signature = gis.generate_signature(target)

pos = 3
for i in range(number):
    tmp = numpy.asarray(map(int, sys.argv[pos+i][1:-1].split(", ")))
    result = gis.normalized_distance(signature, tmp)
    print(result)
