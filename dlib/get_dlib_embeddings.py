######### Extract feature vectors from raw images using Dlib ##################
import face_recognition
import numpy as np
import time
import os

# Path to raw image directory
path = 'raw/'

# path to image files
files = []
# filenames
names = []
embeddings = []

# Get all paths to files and the filenames
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))
            names.append(file)

start = time.time()

# For each raw image
for i in range(len(files)):
    print("[INFO] processing image {}/{}".format(i + 1, len(files)))
    # Load image
    image = face_recognition.load_image_file(files[i])
    # Extract embedding and append it to list
    embeddings.append(face_recognition.face_encodings(image)[0])

print("--- %s seconds ---" % (time.time() - start))

# Save the embeddings are binary files of NumPy format
np.save("embeddings.npy", embeddings)
np.save("names.npy", names)
