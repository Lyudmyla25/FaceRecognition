import numpy as np
import cv2
import torch
import os
from torchvision.models import resnet18
from time import time
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
N_sample = 700
p = 0.08
embedding_size = 25
img_sizes = (1000,1000)
# img1 = cv2.imread("neutral_front/003_03.jpg")
# # img1 = np.resize(img1, img_sizes)
# img = cv2.resize(img1, (1000,1000))
# img2 = cv2.imread("smiling_front/003_08.jpg")
# # img2 = np.resize(img2, img_sizes)
# img3 = cv2.imread("neutral_front/009_03.jpg")
# # img2 = np.resize(img3, img_sizes)
model = resnet18(pretrained=True)

def get_embed(img, model = model):
    '''get resNet mapping of size 1000 for each image'''
    return model(torch.Tensor(img).unsqueeze(0).permute(0,3,1,2)).detach().numpy()

def read_image(path, new_shape = img_sizes):
    '''read and reshape the image'''
    img = cv2.imread(path)
    return cv2.resize(img, new_shape)

start  = time()
lst = os.listdir("neutral_front")
ids = [x[:3] for x in lst if len(x)==10]
photos_by_id = {x:dict({"neutral":get_embed(read_image("neutral_front/"+x+"_03.jpg")),
                        "smiling":get_embed(read_image("smiling_front/"+x+"_08.jpg"))}) for x in ids}
sample_generator = np.random.choice([False, True], N_sample, p = (1-p,p))
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

data_x_train = np.array(emb_diff)[:,0,:]**2
data_y_train = sample_generator.astype(int)
#
# clf  =tree.DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_leaf=10)
# clf.fit(data_x_train, data_y_train)

rf = ensemble.RandomForestClassifier(n_estimators=500,class_weight={0: 1, 1: 100}, max_features=10)
rf.fit(data_x_train, data_y_train)
sample_generator_test = np.random.choice([False, True], 300, p = (1-p,p))
emb_diff = []
for i in list(sample_generator_test):
    if i:
        id1, id2 = np.random.choice(test_ids, 2)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id2]["smiling"]))
    else:
        id1 = np.random.choice(test_ids)
        emb_diff.append(np.abs(photos_by_id[id1]["neutral"] - photos_by_id[id1]["smiling"]))

data_x_test = np.array(emb_diff)[:,0,:]**2
data_y_test = sample_generator_test.astype(int)
# print(clf.score(data_x_train, data_y_train), clf.score(data_x_test, data_y_test))
print(rf.score(data_x_train, data_y_train), rf.score(data_x_test, data_y_test))

allowed_ids = ids[:45] + ids[90:95]
actual_res = np.zeros(len(ids))
actual_res[:45] = 1
actual_res[90:95] = 1
predicted = np.zeros(len(ids))

for i in range(len(ids)):
    current = photos_by_id[ids[i]]["smiling"]
    for id in allowed_ids:
        checked = photos_by_id[id]["neutral"]
        if rf.predict((checked-current)**2)==1:
            predicted[i] = 1
            break


confusion_matrix(actual_res, predicted)