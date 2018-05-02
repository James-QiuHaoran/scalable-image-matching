import time
from image_match.goldberg import ImageSignature
gis = ImageSignature()
# a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
# a = gis.generate_signature('/home/jamesqiu/Desktop/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')

database = []
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/1m.jpg"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/2m.jpg"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/3m.jpg"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/4.JPG"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/5.JPG"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/6.JPG"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/7.JPG"))
#database.append(gis.generate_signature("/home/jamesqiu/Desktop/8.JPG"))

start_time = time.time()
signature = gis.generate_signature("/home/jamesqiu/Desktop/1m.jpg")
print(type(signature))
print(signature)
#start_time = time.time()
# b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
# b = gis.generate_signature('/home/jamesqiu/Desktop/mona-lisa-67506_960_720.jpg')

#test = gis.generate_signature("/home/jamesqiu/Desktop/testm.jpg")
print("--- %s seconds ---" % (time.time() - start_time))
#for x in database:
#    result = gis.normalized_distance(x, test)
#    print(result)

# result = gis.normalized_distance(a, b)
#print("--- %s seconds ---" % (time.time() - start_time))
#print(result)
