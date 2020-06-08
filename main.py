import numpy as np
import cv2
import torch
import os
from torchvision.models import resnet18
from time import time
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
import pdb
import matplotlib.pyplot as plt
import copy
import skimage
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square

N_sample = 700
p = 0.2
embedding_size = 25
img_sizes = (1000, 1000)
model = resnet18(pretrained=True)


def get_embed(img, model=model):
    '''get resNet mapping of size 1000 for each image'''
    return model(torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2)).detach().numpy()


def read_image(path, new_shape=img_sizes):
    '''read and reshape the image'''
    img = cv2.imread(path)
    return cv2.resize(img, new_shape)


def select_face(path, scale=1.06, neigh=5):
    img = read_image(path)
    face = face_cascade.detectMultiScale(img, scale, neigh)
    while len(face) != 1:
        if len(face) == 0:
            neigh -= 1
        else:
            neigh += 1
        face = face_cascade.detectMultiScale(img, scale, neigh)
    x, y, w, h = face[0]
    img_face = img[x:(x + h), y:(y + h)]
    img4 = auto_canny(img_face)
    skel = skimage.morphology.skeletonize(img4 // 255)
    return skel * 255


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


start = time()
# get ids
lst = os.listdir("neutral_front")
ids = [x[:3] for x in lst if len(x) == 10]
#

# use resnet for each photo
photos_by_id = {x: dict({"neutral": get_embed(read_image("neutral_front/" + x + "_03.jpg")),
                         "smiling": get_embed(read_image("smiling_front/" + x + "_08.jpg"))}) for x in ids}
#

# generate set of pairs neutral + smiling
sample_generator = np.random.choice([False, True], N_sample, p=(p, 1 - p))
train_ids = ids[:90].copy()
test_ids = ids[90:].copy()
emb_diff = []
for i in list(sample_generator):
    if i:
        id1, id2 = np.random.choice(train_ids, 2)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id2]["smiling"]))
    else:
        id1 = np.random.choice(train_ids)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id1]["smiling"]))

data_x_train = np.array(emb_diff)[:, 0, :] ** 2
data_y_train = sample_generator.astype(int)
#

# generate table with correct faces: neutral == smiling
ids_table = []
for id in ids:
    diff = np.abs(photos_by_id[id]["neutral"] - photos_by_id[id]["smiling"]) ** 2
    ids_table.append(diff)

ids_table = np.array(ids_table)[:, 0, :]
#

# random forest model
rf = ensemble.RandomForestClassifier(n_estimators=500, class_weight={0: 1, 1: 1}, max_features=10)
rf.fit(data_x_train, data_y_train)
#

# test model
sample_generator_test = np.random.choice([False, True], 300, p=(p, 1 - p))
print(rf.predict(ids_table).mean())

emb_diff = []
for i in list(sample_generator_test):
    if i:
        id1, id2 = np.random.choice(test_ids, 2)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id2]["smiling"]))
    else:
        id1 = np.random.choice(test_ids)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id1]["smiling"]))

data_x_test = np.array(emb_diff)[:, 0, :] ** 2
data_y_test = sample_generator_test.astype(int)
print(rf.score(data_x_train, data_y_train), rf.score(data_x_test, data_y_test))
#

# calculate result table
allowed_ids = ids[:45] + ids[90:95]
actual_res = np.ones(len(ids))
actual_res[:45] = 0
actual_res[90:95] = 0
predicted = np.ones(len(ids))

for i in range(len(ids)):
    current = photos_by_id[ids[i]]["smiling"]
    print(i)
    for id in allowed_ids:
        checked = photos_by_id[id]["neutral"]
        if rf.predict((checked - current) ** 2) == 0:
            predicted[i] = 0
            break

confusion_matrix(actual_res, predicted)
#


face_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, 'haarcascade_eye.xml'))

smile_faces_detection = {id: select_face("smiling_front/" + id + "_08.jpg") for id in ids}
neutral_faces_detection = {id: select_face("neutral_front/" + id + "_03.jpg") for id in ids}

# use resnet for each photo
photos_by_id = {x: dict({"neutral": get_embed(neutral_faces_detection[id]),
                         "smiling": get_embed(smile_faces_detection[id])}) for x in ids}
#

# generate set of pairs neutral + smiling
sample_generator = np.random.choice([False, True], N_sample, p=(p, 1 - p))
train_ids = ids[:90].copy()
test_ids = ids[90:].copy()
emb_diff = []
for i in list(sample_generator):
    if i:
        id1, id2 = np.random.choice(train_ids, 2)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id2]["smiling"]))
    else:
        id1 = np.random.choice(train_ids)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id1]["smiling"]))

data_x_train = np.array(emb_diff)[:, 0, :] ** 2
data_y_train = sample_generator.astype(int)
#

# generate table with correct faces: neutral == smiling
ids_table = []
for id in ids:
    diff = np.abs(photos_by_id[id]["neutral"] - photos_by_id[id]["smiling"]) ** 2
    ids_table.append(diff)

ids_table = np.array(ids_table)[:, 0, :]
#

# random forest model
dt = tree.DecisionTreeClassifier()
dt.fit(data_x_train, data_y_train)
rf = ensemble.RandomForestClassifier(class_weight={0: 3, 1: 1})
rf.fit(data_x_train, data_y_train)
print(dt.predict(data_x_train).mean())
#

# test model
sample_generator_test = np.random.choice([False, True], 300, p=(p, 1 - p))
print(rf.predict(ids_table).mean())

emb_diff = []
for i in list(sample_generator_test):
    if i:
        id1, id2 = np.random.choice(test_ids, 2)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id2]["smiling"]))
    else:
        id1 = np.random.choice(test_ids)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id1]["smiling"]))

data_x_test = np.array(emb_diff)[:, 0, :] ** 2
data_y_test = sample_generator_test.astype(int)
print(rf.score(data_x_train, data_y_train), rf.score(data_x_test, data_y_test))
#

# calculate result table
allowed_ids = ids[:45] + ids[90:95]
actual_res = np.ones(len(ids))
actual_res[:45] = 0
actual_res[90:95] = 0
predicted = np.ones(len(ids))

for i in range(len(ids)):
    current = photos_by_id[ids[i]]["smiling"]
    print(i)
    for id in allowed_ids:
        checked = photos_by_id[id]["neutral"]
        if rf.predict((checked - current) ** 2) == 0:
            predicted[i] = 0
            break

confusion_matrix(actual_res, predicted)
