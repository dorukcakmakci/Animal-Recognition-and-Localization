'''
    Authors    : Doruk Çakmakçı, İrem Ural
    Description: Feature Extraction of training images and training classifiers on the resulting features
    Date       : 09/01/2019
'''

import numpy as np
import PIL
import os
import torch
import pdb
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from PIL import Image
from resnet import resnet50
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

root_dir = "../data/train"
categories = ["n01615121",
              "n02099601",
              "n02123159",
              "n02129604",
              "n02317335",
              "n02391049",
              "n02410509",
              "n02422699",
              "n02481823",
              'n02504458']

# use pretrained resnet50 residual NN in eval mode.
model = resnet50(pretrained = True)
model.eval()

index = 1
features = []

# traverse all images and extract features using NN.
for category in categories:

    partial_path = os.path.join(root_dir, category)
    current_files = os.listdir(partial_path)

    for file_name in current_files:

        image_path = os.path.join(partial_path, file_name)

        if image_path.endswith("a.JPEG"):
            continue

        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        width = image.shape[0]
        height = image.shape[1]
        result = False
        flag = False

        # insert padding to the images
        if width - height > 0:
            flag = True

        if flag:
            diff = width - height

            if diff % 2 == 0:
                pad_width = int(diff / 2)
                padding = np.zeros((width, pad_width,3))
                result = np.concatenate((padding, image, padding), axis=1)
            else:
                pad_width = int(diff / 2)
                padding_left = np.zeros((width, pad_width,3))
                padding_right = np.zeros((width, pad_width + 1,3))
                result = np.concatenate((padding_left, image, padding_right), axis=1)

        else:
            diff = height - width

            if diff % 2 == 0:
                pad_width = int(diff / 2)
                padding = np.zeros((pad_width, height,3))
                result = np.concatenate((padding, image, padding), axis=0)
            else:
                pad_width = int(diff / 2)
                padding_top = np.zeros((pad_width, height,3))
                padding_bottom = np.zeros((pad_width+1, height,3))
                result = np.concatenate((padding_top, image, padding_bottom), axis=0)

        # resize images to 224x24
        result = Image.fromarray(result.astype('uint8'), 'RGB')
        result = result.resize((224,224), PIL.Image.NEAREST)      
        result = np.asarray(result)

        # normalize images
        result = result / 255
        result[:, :, 0] = ( result[:, :, 0] - 0.485 ) / 0.229
        result[:, :, 1] = ( result[:, :, 1] - 0.456 ) / 0.224
        result[:, :, 2] = ( result[:, :, 2] - 0.406 ) / 0.225
        result = result.astype('float32')

        #save images
        assert result[0,0].dtype == 'float32' 
        temp = Image.fromarray(result.astype('uint8'), 'RGB')
        image_path = image_path.split('.JPEG')[0] + 'a.JPEG'
        temp.save(image_path)

        # feature extraction via resnet50
        image = np.reshape(result, [1, 224, 224, 3])
        image = image.transpose([0, 3, 1, 2])
        image = torch.from_numpy(image)
        feature_vector = model(image)
        feature_vector = feature_vector.detach().numpy()

        # l2-normalize feature vectors
        feature_vector = feature_vector / ( np.linalg.norm(feature_vector) + 0.0001 )
        feature_vector = np.append(feature_vector, index)
        features.append(feature_vector.tolist())
        
    index += 1

#dataset = np.random.shuffle(features)
dataset = np.asarray(features)

labels = dataset[:, -1]
features = dataset[:, 0:-1]

# define parameters and their values to optimize
parameters = [{
                'kernel': ['rbf', 'linear', 'poly'], 
                'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                'C': [0.01, 0.1, 1, 10, 100, 1000]},
]
# train optimal SVM models using Grid Search
svm_models = []
# One versus all approach is used while simulating multi-class SVM
for i in range(1,11):
    current_labels = np.copy(labels)
    for j in range(len(labels)):
        if labels[j] != i:
            current_labels[j] = 0
        else:
            current_labels[j] = 1

    # use grid search to determine optimal parameters  for SVM model 
    print("# Tuning hyper-parameters for class %d" % i)
    print()

    clf = GridSearchCV(SVC(), parameters, cv=5)
    clf.fit(features, current_labels)
    svm_models.append(clf.best_estimator_)

    # save the model to disk
    filename = 'svm_' + str(i)+ '.sav'
    root = '../models/'
    path = os.path.join(root, filename)
    pickle.dump(clf.best_estimator_, open(path, 'wb'))

    # find best parameters among the specified parameter set
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
    print()






        
        




