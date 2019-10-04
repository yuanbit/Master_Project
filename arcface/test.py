import face_model
import argparse
import cv2
import sys
import numpy as np
import time
import os

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='models/model-r100-ii/model, 0', help='path to load model.')
parser.add_argument('--ga-model', default='models/model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

path = 'datasets/data1_mtcnnpy_112'

model = face_model.FaceModel(args)

files = []
names = []
embeddings = []

#r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))
            names.append(file)
print("\n \n")

start = time.time()
for i in range(len(files)):

    print("[INFO] processing image {}/{}".format(i + 1, len(files)))

    img = cv2.imread(files[i])
    img = model.get_input(img)
    embeddings.append(model.get_feature(img))

print("--- %s seconds ---" % (time.time() - start))

np.save("arcface_embeddings.npy", embeddings)
np.save("arcface_names.npy", names)


# #print(f1[0:10])
# # gender, age = model.get_ga(img)
# print("gender {}".format(gender))
# # print(age)
# #sys.exit(0)
# img2 = cv2.imread('Roger_Maltbie_golf.png')
# img2 = model.get_input(img2)
# f2 = model.get_feature(img2)
#
# print(f2)
#
# print(f2)
# dist = np.sum(np.square(f1-f2))
# print("dist {}".format(dist))
# sim = np.dot(f1, f2.T)
# print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
