import face_recognition
import numpy as np
import time
import os

path = 'raw/'

files = []
names = []
embeddings = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))
            names.append(file)

start = time.time()
for i in range(len(files)):

    print("[INFO] processing image {}/{}".format(i + 1, len(files)))

    image = face_recognition.load_image_file(files[i])
    embeddings.append(face_recognition.face_encodings(image)[0])

print("--- %s seconds ---" % (time.time() - start))

np.save("data1_embeddings.npy", embeddings)
np.save("data1_names.npy", names)
