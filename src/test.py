import numpy as np
import PIL
import os
import torch
import pdb
import pickle
import operator
from PIL import Image
from resnet import resnet50

root_svm = '../models/'
svm = [ "svm_1.sav",
        "svm_2.sav",
        "svm_3.sav",
        "svm_4.sav",
        "svm_5.sav",
        "svm_6.sav",
        "svm_7.sav",
        "svm_8.sav",
        "svm_9.sav",
        "svm_10.sav"]

image_map = [0] * 100

models = []
for name in svm:
    models.append(pickle.load(open(os.path.join(root_svm, name), 'rb')))

image_root = '../data/test/windows'
image_dirs = []
for i in range(0,100):
    image_dirs.append(str(i))
window_names = []
for i in range(1,51):
    window_names.append(str(i) + '.JPEG')

features = []
#model = resnet50()
model = resnet50(pretrained = True)
model.eval()

index = 0

for image_dir in image_dirs:
    features = []
    for name in window_names:
        path = os.path.join(image_root, image_dir, name)
        if path.endswith("a.JPEG"):
            continue
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)

        #print(image.shape)
        width = image.shape[0]
        height = image.shape[1]
        result = False
        flag = False
        if width - height > 0:
            flag = True
        if flag:
            diff = width - height
            if diff % 2 == 0:
                pad_width = int(diff / 2)
                padding = np.zeros((width, pad_width,3))
                #print(padding.shape)
                #print(image.shape)
                result = np.concatenate((padding, image, padding), axis=1)
                #print(result.shape)
            else:
                pad_width = int(diff / 2)
                padding_left = np.zeros((width, pad_width,3))
                padding_right = np.zeros((width, pad_width + 1,3))
                result = np.concatenate((padding_left, image, padding_right), axis=1)
               # print(result.shape)
        else:
            diff = height - width
            if diff % 2 == 0:
                pad_width = int(diff / 2)
                padding = np.zeros((pad_width, height,3))
               # print(padding.shape)
               # print(image.shape)
                result = np.concatenate((padding, image, padding), axis=0)
                #print(result.shape)
            else:
                pad_width = int(diff / 2)
                padding_top = np.zeros((pad_width, height,3))
                padding_bottom = np.zeros((pad_width+1, height,3))
                result = np.concatenate((padding_top, image, padding_bottom), axis=0)
                #print(result.shape)

        result = Image.fromarray(result.astype('uint8'), 'RGB')
        result = result.resize((224,224), PIL.Image.NEAREST)
        #print(result.size)        

        result = np.asarray(result)
        result = result / 255
        result[:, :, 0] = ( result[:, :, 0] - 0.485 ) / 0.229
        result[:, :, 1] = ( result[:, :, 1] - 0.456 ) / 0.224
        result[:, :, 2] = ( result[:, :, 2] - 0.406 ) / 0.225
        result = result.astype('float32')
        #print(result.dtype)

        assert result[0,0].dtype == 'float32' 
        temp = Image.fromarray(result.astype('uint8'), 'RGB')
        image_path = path.split('.JPEG')[0] + 'a.JPEG'
        temp.save(image_path)

        # start feature extraction
        image = np.reshape(result, [1, 224, 224, 3])
        #print(image.dtype)
        image = image.transpose([0, 3, 1, 2])
        image = torch.from_numpy(image)
        feature_vector = model(image)
        #pdb.set_trace()
        feature_vector = feature_vector.detach().numpy()
        feature_vector = feature_vector / ( np.linalg.norm(feature_vector) + 0.0001 )
        features.append(feature_vector.tolist())
    dataset = np.asarray(features)
    
    # test
    acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for data in dataset:
        for i in range(10):
            pred = models[i].predict(data)
            if pred == 1:
                acc[i] += 1
        
    predicted_class, value = max(enumerate(acc), key=operator.itemgetter(1))
    image_map[index] = predicted_class
    print()
    print('predicted class for image ' + image_dir + ' is ' + str(predicted_class))
    print()
    index += 1

# preprocess candidate windows


# loaded_model.score(X_test, Y_test)


