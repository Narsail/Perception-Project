Project: Perception Pick & Place
---

## Object Recognition

The object recognition [file][project] is split into 3 parts:

* the algorithms of the perception exercise 2
* the algorithms of the perception exercise 3
* testing the algorithm on different scenarios

### Exercise 2 Part

This part represents various filter algorithms to prepare the cloud point data for the object recognition. 
All filters except the statistical outlier filter have been applied in an exercise before but have been tweaked for this application. 
In the following subsections i will go through every filter to explain its application:

Comment: The filters were applied according to the explanation order.

#### Statistical Outlier filter 

Is used to filter out random noise. For this it analyses a set number of neighbouring points and their distance to the chosen point. 
If the average distance exceeds a certain threshold, that point will be filtered out.

#### Voxel Downsampling

This filter is used due to performance reasons. The provided point cloud data is too extensive to use in a feasible time. Therefore we downsample the data.

#### Passthrough filter

If we know the location of the desired objects we can mask the view of vision to exclude all unnecessary information. 
We used the z-axis filtering to remove the table parts except where the objects are placed on. 
Furthermore we utilized a y-axis filter to remove the outmost left and right part of the table which were empty.

#### RANSAC

The RANSAC filter is able to filter / extract specific geometries. In this case the filter is set on the table based on the PLANE geometry. 
In this way it was possible the separate the table from the objects into the point clouds.

The following images show the ransac results:
![alt][RANSAC_table]
![alt][RANSAC_objects]

#### Euclidean clustering 

With euclidean clustering we can assign points in a point cloud to labels based on their distance to each other. 
By defining minimum and maximum values it will find independently a certain number of clusters.

With this algorithm with can split the cloud points into objects to give them independently into the object recognition algorithm.

![alt][clustering]


To find the best values for every filter step i used the world 3 example and added the filters step by step to observe the result and tweak accordingly.

### Exercise 3 Part

Exercise 3 introduced the object recognition algorithm - a [support vector machine](https://de.wikipedia.org/wiki/Support_Vector_Machine). It is used to separate objects based on features. 
It has to be trained with pre recorded samples and assigned labels.

The values used in this part have been tweaked in the same way as in exercise 2, mostly through playing around (trial and error).

To recognize the objects it had to fulfill the following steps:

1. record enough trainings data
2. train a svm to separate the data points
3. add the prediction step into the pipeline

To record the trainings data i had to slightly modify [features.py][features] and [capture_features.py][capture] to add custom bins and the number of samples per object needed for the training.
Then i trained the SVM via [train_svm.py][train], but with a rbf kernel to separate non linear distributions, which resulted in the following confusion matrices:

![alt][confusion]
![alt][confusion_norm]

The average accuracy is around 94% which was enough to fulfill the requirements of this project.

![alt][accuracy]

### Run the test worlds

To complete the project the pipeline had to classify 100% of the objects in world 1, 80% of world 2 and 75% of world 3. 

#### World 1

Output: [Output 1][output_1]

Accuracy: 100%

The following visual output was generated in world 1:

![alt][test_1_recognition]

#### World 2

Output: [Output 2][output_2]

Accuracy: 100%

The following visual output was generated in world 2:

![alt][test_2_recognition]

#### World 3

Output: [Output 3][output_3]

Accuracy: 88% (7 out of 8)

The following visual output was generated in world 3:

![alt][test_3_recognition]

## Project discussion

### What worked

The implementation of the object recognition was straight forward thanks to the exercises we had to complete before starting the project. 
With adding the filters step by step and tweaking the values accordingly i didn't encounter any major issue. 
I'm only a bit constraint on the time i can use on that project so that i fall back to the minimum requirements.

### Future Work

There is definitely a bit room of improvement for the object recognition itself - the svm. We could experiment with more object positions, a richer feature set as well as with data augmentation (translated, resized, color-customized).

The second area would be to improve the noise filtering and the clustering to get the last object on the world 3 table which is partly hidden by the book. My current implementation has its difficulties to see that object.







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
[accuracy]: ./images/accuracy.png
[RANSAC_table]: ./images/RANSAC_table.png
[RANSAC_objects]: ./images/RANSAC_objects.png
[clustering]: ./images/clustering.png
[test_1_recognition]: ./images/test_1_object_recognition.png
[test_2_recognition]: ./images/test_2_object_recognition.png
[test_3_recognition]: ./images/test_3_object_recognition.png