Project: Perception Pick & Place
---

## Object Recognition

The object recognition [file][project] is split into 3 parts:

* the algorithms of the perception exercise 2
* the algorithms of the perception exercise 3
* storing the results in a yml file

### Exercise 2 Part

This step includes the following filter steps:

* statistical outlier filter
* voxel downsampling
* Passthrough filter (y and z axis)
* RANSAC filter
* euclidean clustering

To find the best values for every filter step i used the world 3 example and added the filters step by step to observe the result and tweak accordingly.

The following images show the ransac results:
![alt][RANSAC_table]
![alt][RANSAC_objects]

as well as the result after the clustering step:

![alt][clustering]

### Exercise 3 Part

The values used in this part have been tweaked in the same way as in exercise 2, mostly through playing around (trial and error).

To recognize the objects i had to fulfill the following steps:

1. record enough trainings data
2. train a svm to separate the data points
3. add the prediction step into the pipeline

To record the trainings data i had to slightly modify [features.py][features] and [capture_features.py][capture] to add custom bins and the number of samples per object needed for the training.
Then i trained the SVM via [train_svm.py][train], but with a rbf kernel to separate non linear distributions, which resulted in the following confusion matrices:

![alt][confusion]
![alt][confusion_norm]

The average accuracy is around 93% which was enough to fulfill the requirements of this project.

### Storing in yaml files

With this setup we ran all three worls which resulted in the following yaml files:

* World 1: [Output 1][output_1]
* World 2: [Output 2][output_2]
* World 3: [Output 3][output_3]  

[//]: # (File References)

[features]: ./source_files/features.py
[capture]: ./source_files/capture_features_pr2_robot.py
[train]: ./source_files/train_svm.py
[project]: ./source_files/project_template.py

[output_1]: ./output_1.yaml
[output_2]: ./output_2.yaml
[output_3]: ./output_3.yaml

[//]: # (Image References)

[confusion]: ./images/confusion.png
[confusion_norm]: ./images/confusion_norm.png
[RANSAC_table]: ./images/RANSAC_table.png
[RANSAC_objects]: ./images/RANSAC_objects.png
[clustering]: ./images/clustering.png
[test_1_recognition]: ./images/test_1_object_recognition.png
[test_2_recognition]: ./images/test_2_object_recognition.png
[test_3_recognition]: ./images/test_3_object_recognition.png