import numpy as np
import cv2
import torch
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
import pickle
import torchvision

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
img_sizes = (1000, 1000)
p = 0.15
N_sample = 600
norm = 0.5


def get_embed(img, model=model, lst=True):
    '''get resNet mapping of size 5 for each image'''
    if lst:
        result = model([torch.Tensor(x).permute(2, 0, 1) / 255.0 for x in img])
    else:
        result = model([torch.Tensor(img).permute(2, 0, 1) / 255.0])
    return [x["keypoints"][0][:5].detach().numpy()[:, :2] for x in result]


def read_image(path, new_shape=img_sizes):
    '''read and reshape the image'''
    img = cv2.imread(path)
    return cv2.resize(img, new_shape)


def normalized_dist_between_points(points):
    dist = pairwise_distances(points)
    dist /= dist[1, 2]  # divide by distance between eyes
    return dist[np.triu_indices(5, 1)]


def generate_pair_diff_sample(n2, s2, p=p, norm=2, N_sample=600):
    sample_generator = np.random.choice([False, True], N_sample, p=(p, 1 - p))
    emb_diff = []
    for i in list(sample_generator):
        i1, i2 = np.random.choice(np.arange(0, s2.shape[0]), 2)
        if i:
            # generate sample for different faces
            emb_diff.append(np.abs(n2[i1] - s2[i2]))
        else:
            # generate sample for same faces
            emb_diff.append(np.abs(n2[i1] - s2[i1]))

    data_x_train = np.array(emb_diff) ** norm
    data_y_train = sample_generator.astype(int)
    return data_x_train, data_y_train


def skutecznosc(m):
    return np.diag(m).sum() / m.sum()


# get ids
lst = os.listdir("neutral_front")
ids = [x[:3] for x in lst if len(x) == 10]
#

smile_faces_img = [read_image("smiling_front/" + id + "_08.jpg") for id in ids]
neutral_faces_img = [read_image("neutral_front/" + id + "_03.jpg") for id in ids]

# s1 = get_embed(smile_faces_img)
# n1 = get_embed(neutral_faces_img)

###run for s1 and n1: smiling faces and neutral faces
# s1 = []
# for i,id in enumerate(ids):
#     print(i)
#     s1.append(get_embed([smile_faces_img[i]]))
#
# n1 = []
# for i,id in enumerate(ids):
#     print(i)
#     n1.append(get_embed([neutral_faces_img[i]]))
#
# with open('s1.pickle', 'wb') as handle:
#     pickle.dump(s1, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('n1.pickle', 'wb') as handle:
#     pickle.dump(n1, handle, protocol=pickle.HIGHEST_PROTOCOL)
#####

# load smiling faces s1 and neutral faces n1
with open('s1.pickle', 'rb') as handle:
    s1 = pickle.load(handle)

with open('n1.pickle', 'rb') as handle:
    n1 = pickle.load(handle)
#


s2 = [normalized_dist_between_points(x[0]) for x in s1]
s2 = np.array(s2)
n2 = [normalized_dist_between_points(x[0]) for x in n1]
n2 = np.array(n2)
# generate set of pairs neutral + smiling

train_index = 90

data_x_train, data_y_train = generate_pair_diff_sample(n2[:train_index, :], s2[:train_index, :], 0.45, norm, N_sample)
data_x_test, data_y_test = generate_pair_diff_sample(n2[train_index:, :], s2[train_index:, :], 0.5, norm, N_sample)
clf = LogisticRegression(random_state=0).fit(data_x_train, data_y_train)
y_pred = clf.predict(data_x_test)

# calculate result score
skutecznosc(confusion_matrix(data_y_test, y_pred))
clf.score(data_x_test, data_y_test)

# sample of points on image
i=5
id = ids[i]

##smile face
img = read_image("smiling_front/"+id+"_08.jpg")
points = s1[i][0]
img1 = img.copy()

# Radius of circle
radius = 2
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
for x,y in points:
    img1 = cv2.circle(img1, (x,y), radius, color, thickness)

cv2.imwrite("img.png", img1)
##
