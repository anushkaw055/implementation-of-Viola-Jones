# Implementation-of-Viola-Jones

Implement the Viola-Jones Algorithm for rapid face detection in python from scratch. First developed a feature extraction script, which extracted 2.5 thousand features from a 19 by 19 grayscale image. I applied the feature extraction script to a 2000 image of non-faces and 500 images of faces. I implemented the AdaBoost algorithm through the python multiprocessor module, leading to a decrease in execution time by 20%. I ran 10 rounds of the algorithm to achieve an empirical error of 67% on the testing data set. Feature manipulated the cost function on the algorithm to priories false-positive, which led to a 5.4% false-positive error.


I have followed the algorithm given in the paper and have implemented the technique of non-maximal suppression as well to remove the overlapping windows or square boxes from appearing.
