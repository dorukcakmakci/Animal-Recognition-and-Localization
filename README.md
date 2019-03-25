# Animal Recognition and Localization

This project includes a multi-class SVM to classify animal images and object localization code to locate the classified animal frames in the images. 

The dataset used for training and testing the classifier is a small subset of ImageNet(400 images for training and 100 images for testing. Train and test datasets are uniform in terms of animal class).

The train and test set images contain images of eagle, dog, cat, tiger, starfish, zebra, bison, antilope, monkey and elephant. The multiclass SVM implementation uses 10 different One-Vs-All SVMs to determine the class of the input image. The hyperparameters of SVMs(kernel type, gamma, C) are optimized using Grid-Search. The feature extraction from the images is done by the ResNet-50 network(pre-trained on the ImageNet dataset in order to extract visual features) and SVMs classify the images based on these features. PyTorch is used as the deep learning framework.

The candidate windows for object localization are extracted using Edge Boxes method. The MATLAB implementation of the technique is used. The Edge Boxes implementation can be found in https://github.com/pdollar/edges. However, before using this implementation, Piotr's MATLAB Toolbox must be installed must be installed from https://pdollar.github.io/toolbox/. 

For quantitative performance evaluation, classification accuracy and localization accuracy are used. Precision and recall for each object type and confusion matrix are computed. For localization accuracy, overlap ratio is used.

For further implementation details and results, refer to "report.pdf".

 
