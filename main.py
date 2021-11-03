# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
#%%
import cv2
import numpy as np
import os
import tensorflow as tf
import json
import pickle

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor
extractor = HandShapeFeatureExtractor().get_instance()
# %%
# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
from frameextractor import frameExtractor

# Hardcoded label information for training data
train_label_output = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7", "8":"8", "9":"9", "DecreaseFanSpeed":"10", "FanOff":"11", "FanOn":"12",
          "IncreaseFanSpeed":"13", "LightOff":"14", "LightOn":"15", "SetThermo":"16"}

if not os.path.exists("./train_features.pk") or not os.path.exists("./train_labels.pk"):
  train_frame_path = "./trainfeatures/"
  if not os.path.exists(train_frame_path):
      os.mkdir(train_frame_path)

  trainvid_path = "./traindata/"
  for path in os.listdir(trainvid_path):
      frameExtractor(os.path.join(trainvid_path, path), os.path.join(train_frame_path, path.split("_PRACTICE_")[0]),int(path[-11:][0])-1)

  train_features = []
  train_labels = []
  for path in os.listdir(train_frame_path):
      for i in range(1,5):
        # preprocess image
          img = cv2.cvtColor(cv2.imread(os.path.join(train_frame_path, path, f"0000{i}.png")), cv2.COLOR_RGB2GRAY)
          # img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
          train_features.append(extractor.extract_feature(img))
          train_labels.append(path)
  print("Train features extracted")

  with open("train_features.pk", 'wb') as f:
    pickle.dump(train_features, f)
  with open("train_labels.pk", "wb") as f:
    pickle.dump(train_labels, f)
# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
if not os.path.exists("./test_features.pk") or not os.path.exists("./test_labels.pk"):
  test_frame_path = "./testfeatures/"
  if not os.path.exists(test_frame_path):
      os.mkdir(test_frame_path)

  testvid_path = "./test/"
  for path in os.listdir(testvid_path):
      frameExtractor(os.path.join(testvid_path, path), os.path.join(test_frame_path, path[2:-4]), 0)

  test_features = []
  test_labels = []
  for path in os.listdir(test_frame_path):
      test_features.append(extractor.extract_feature(cv2.cvtColor(cv2.imread(os.path.join(test_frame_path, path, f"00001.png")), cv2.COLOR_RGB2GRAY)))
      test_labels.append(path[3:])
  print("Test features extracted")

  with open("test_features.pk", 'wb') as f:
    pickle.dump(test_features, f)
  with open("test_labels.pk", "wb") as f:
    pickle.dump(test_labels, f)


# %%
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
with open("train_features.pk", "rb") as f:
  train_features = pickle.load(f)
with open("train_labels.pk", "rb") as f:
  train_labels = pickle.load(f)
with open("test_features.pk", "rb") as f:
  test_features = pickle.load(f)
with open("test_labels.pk", "rb") as f:
  test_labels = pickle.load(f)

cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)
# %%
preds = {}
for i, Xtest in enumerate(test_features):
  similarities = {key:0 for key in train_labels}
  n = 0
  for j, Xtrain in enumerate(train_features):
    # similarities[train_labels[j]] = 1 - cosine_similarity(Xtrain, Xtest).numpy()
    similarities[train_labels[j]+"_"+str(j)] = np.dot(Xtrain, np.transpose(Xtest)
                                                      )/(np.linalg.norm(Xtrain)*np.linalg.norm(Xtest)) # np.abs(cosine_similarity(Xtrain, Xtest).numpy())# 
    n += 1
  
  # # getting average cosine distance for test vs training set
  # for key in similarities:
  #   similarities[key] /= n
    
  preds[test_labels[i]] = {k:v for k,v in sorted(similarities.items(), 
                            key=lambda x: x[1], reverse=True)}

# Accuracy
total_correct = 0
with open('results.csv','w') as f:
  lines = [f"{key}\t{train_label_output[list(preds[key].keys())[0].split('_')[0]]}" for key in preds]
  f.writelines(lines)
