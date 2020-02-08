import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import cv2
import random

faces_path_train ='./dataset/trainset/faces/'
non_faces_path_train = './dataset/trainset/non-faces/'
faces_path_test = './dataset/testset/faces/'
non_faces_path_test = './dataset/testset/non-faces/'
faces_size = 500
non_faces_size = 2500

class FaceDetector:
    def __init__(self, NoOfAdaboost = 20):
        self.NoOfAdaboost = NoOfAdaboost
        self.alphas = []
        self.infos = []
        self.classifiers = []

    def Train(self, Training, NoOfFaces, NoOfNonFaces, window_size):
        print('Executing training code')
        weights = np.zeros(len(Training))
        training_data = []

        for x in range(len(Training)):
            training_data.append((getIntegralImage(Training[x][0]), Training[x][1]))
            if Training[x][1] == 1:                
                weights[x] = 1.0 / (2 * NoOfFaces)
            else:
                weights[x] = 1.0 / (2 * NoOfNonFaces)

        features = self.BuildFeatures(training_data[0][0].shape, window_size)
        X, y, z = self.apply_features(features, training_data)
        save = (features, X, y, z)

        with open("./models/features.pkl", 'wb') as f:
            pickle.dump(save, f)
        
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]

        for t in range(self.NoOfAdaboost):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.TrainWeak(X, y, z, features, weights)
            best_threshold, best_feature, clf, error, accuracy, best_fp, best_fn = self.SelectBest(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
   
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
                
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.classifiers.append(clf)
            accuracy = (len(accuracy) - sum(accuracy))
            self.infos.append((best_feature, best_threshold, (accuracy, len(training_data))))

            print(' Round', str(t + 1))
            print('Alpha : {:.2}'.format(alpha))
            print('Error : {:.2%}'.format(error))
            print('Threshold : {:.2}'.format(best_threshold))
            print('Accuracy : {:.2%}'.format(accuracy/len(training_data)), '(', accuracy, '/', len(training_data),')')
            print('False Postive : {:.2%}'.format(best_fp/len(training_data),), '(', best_fp, '/', len(training_data),')')
            print('False Negative : {:.2%}'.format(best_fn/len(training_data),), '(', best_fn, '/', len(training_data),')')

    def TrainWeak(self, X, y, z, features, weights):
        print('Training weak classifiers')
        total_pos, total_neg = 0, 0

        for w, label in zip(weights, y):
            if label == 1:                
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        for index, feature in enumerate(X):    
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = z[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = Classifier(best_feature, best_threshold, best_polarity)
            
            classifiers.append(clf)

        return classifiers
                
    def BuildFeatures(self, image_shape, window_size):
        imgHeight, imgWidth = image_shape
        features = []
        print('Feature building')
        nums=[]

        featureTypes=[("2V",(1,2)),("2H",(2,1)),
                    ("3H",(3,1)),("3V",(1,3)),("4",(2,2))]

        for type,size in featureTypes:
            num=0

            for width in range(size[0],window_size+1,size[0]):
                for height in range(size[1],window_size+1,size[1]):
                    for x in range(1, imgWidth-width+1):
                        for y in range(1, imgHeight-height+1):
                            features.append(Box(type,x,y,width,height))
                            num+=1
            nums.append(num)

        print('Total features', imgHeight - 1, 'x' ,imgWidth - 1, 'image for window size of', window_size, ' : ', sum(nums))
        
        return np.array(features)

    def SelectBest(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        print('Selecting best classifier from '+ str(len(classifiers)))
        i = 1

        for classifier in classifiers:
            if i % (len(classifiers)//4) == 0:
                print('{:.2%}'.format(i/len(classifiers)), 'finished, please wait...')
            error, accuracy = 0, []
            fp = 0
            fn = 0
            for data, w in zip(training_data, weights):
                classify, feature, feature_value, threshold = classifier.classify(data[0])
                correctness = abs(classify - data[1])
                if data[1] == 0 and classify == 1:
                    fp += 1
                if data[1] == 1 and classify == 0:
                    fn += 1
                accuracy.append(correctness)
                error += w * correctness

            error = error / len(training_data)            
            lbd = 0.8

            if error < best_error and error != 0:
                best_threshold, best_feature, best_clf, best_error, best_accuracy, best_fp, best_fn = threshold, feature, classifier, error, accuracy, fp, fn
            i += 1 
        return best_threshold, best_feature, best_clf, best_error, best_accuracy, best_fp, best_fn
    
    def apply_features(self, features, training_data):
        X = np.zeros((len(features), len(training_data)))
        z = [None] * len(features)
        y = np.array(list(map(lambda data: data[1], training_data)))
        print('Applying features in training set')

        for i in range(len(training_data)):
            print('Features calculation on data : ', i, '/', len(training_data))
            for j in range(len(features)):
                X[j][i] = features[j].compute_feature(training_data[i][0])[1]
                z[j] = features[j].compute_feature(training_data[i][0])[0]
        return X, y, z

    def classify(self, image):
        total = 0
        integralImg = getIntegralImage(image)
        for alpha, clf in zip(self.alphas, self.classifiers):
            total += alpha * clf.classify(integralImg)[0]
        return 1 if total >= 0.6 * sum(self.alphas) else 0

    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:    
            pickle.dump(self, f)

    @staticmethod
    def Load(filename):
        with open(filename+".pkl", 'rb') as f:    
            return pickle.load(f)


class Box:
    def __init__(self, type, x, y, width, height):
        self.type = type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, integralImg):
        result = None
        if self.type == '2V':
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width - 1)
            C = (self.y + self.height//2 - 1, self.x - 1)
            D = (self.y + self.height//2 - 1, self.x + self.width - 1)
            E = (self.y + self.height - 1, self.x - 1)
            F = (self.y + self.height - 1, self.x + self.width - 1)
            result = 2 * integralImg[D[0]][D[1]] + integralImg[A[0]][A[1]] - integralImg[B[0]][B[1]] - 2 * integralImg[C[0]][C[1]] + integralImg[E[0]][E[1]] - integralImg[F[0]][F[1]]

        elif self.type == '2H':
           
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width//2 - 1)
            C = (self.y - 1 , self.x + self.width - 1)
            D = (self.y  + self.height - 1, self.x - 1)
            E = (self.y  + self.height - 1 , self.x + self.width//2 - 1)
            F = (self.y  + self.height - 1 , self.x + self.width - 1)
            result = 2 * integralImg[B[0]][B[1]] + integralImg[F[0]][F[1]] - integralImg[C[0]][C[1]] - 2 * integralImg[E[0]][E[1]] + integralImg[D[0]][D[1]] - integralImg[D[0]][D[1]]

        elif self.type == '3H':
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width//3 - 1)
            C = (self.y - 1 , self.x + self.width//3 * 2 - 1)
            D = (self.y - 1 , self.x + self.width - 1)
            E = (self.y  + self.height - 1 , self.x - 1)
            F = (self.y  + self.height - 1 , self.x + self.width//3 - 1)
            G = (self.y  + self.height - 1, self.x + self.width//3 * 2 - 1)
            H = (self.y  + self.height - 1, self.x + self.width - 1)
            result = 2 * integralImg[B[0]][B[1]] + 2 * integralImg[G[0]][G[1]] - 2 * integralImg[C[0]][C[1]] - 2 * integralImg[F[0]][F[1]] - integralImg[H[0]][H[1]] - integralImg[A[0]][A[1]] + integralImg[D[0]][D[1]] + integralImg[E[0]][E[1]]
        
        elif self.type == '3V':
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width - 1)
            C = (self.y + self.height//3 - 1 , self.x - 1)
            D = (self.y + self.height//3 - 1 ,  self.x + self.width - 1)
            E = (self.y  +  self.height//3 * 2 - 1 , self.x - 1)
            F = (self.y  + self.height//3 * 2 - 1 , self.x + self.width - 1)
            G = (self.y  + self.height - 1, self.x - 1)
            H = (self.y  + self.height - 1, self.x + self.width - 1)
            result = 2 * integralImg[C[0]][C[1]] + 2 * integralImg[F[0]][F[1]] - 2 * integralImg[D[0]][D[1]] - 2 * integralImg[E[0]][E[1]] - integralImg[H[0]][H[1]] - integralImg[A[0]][A[1]] + integralImg[B[0]][B[1]] + integralImg[G[0]][G[1]]

        elif self.type == '4':
            A = (self.y - 1, self.x - 1)
            B = (self.y - 1 , self.x + self.width//2 - 1)
            C = (self.y - 1 , self.x + self.width - 1)
            D = (self.y + self.height//2 - 1, self.x - 1)
            E = (self.y + self.height//2 - 1, self.x + self.width//2 - 1)
            F = (self.y + self.height//2 - 1, self.x + self.width - 1)
            G = (self.y + self.height - 1, self.x - 1)
            H = (self.y + self.height - 1, self.x + self.width//2 - 1)
            I = (self.y + self.height - 1, self.x + self.width - 1)
            result = -integralImg[A[0]][A[1]] + 2 * integralImg[B[0]][B[1]] - integralImg[C[0]][C[1]] + 2 * integralImg[D[0]][D[1]] - 4 * integralImg[E[0]][E[1]] + 2 * integralImg[F[0]][F[1]] - integralImg[G[0]][G[1]] + 2 * integralImg[H[0]][H[1]] - integralImg[I[0]][I[1]]
        return((self.type, self.x, self.y, self.width, self.height), result)

class Classifier:
    def __init__(self, feature, threshold, polarity):
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
        feature = Box(self.feature[0], self.feature[1], self.feature[2], self.feature[3], self.feature[4])
        feature_value = feature.compute_feature(x)[1]
        if self.polarity * feature_value < self.polarity * self.threshold:
            return (1, self.feature, feature_value, self.threshold)
        else:
            return (0, self.feature, feature_value, self.threshold)
    
def getIntegralImage(img):
    row = len(img)
    col = len(img[0])
    integral = np.zeros((row + 1,col +1))
    
    for i in range(1,row + 1):
        for j in range(1,col +1):
            integral[i][j] = int(img[i-1][j-1])
            if i-1 >=0 and j-1 >=0:
                integral[i][j] = integral[i][j] + integral[i-1][j] + integral[i][j-1] + - integral[i-1][j-1] 
            elif i-1 >= 0:
                integral[i][j] = integral[i][j] + integral[i-1][j]
            elif j-1 >= 0:
                integral[i][j] = integral[i][j] + integral[i][j-1]
            
    return integral

def getImage(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return image

def normalize(img):
    img = np.array(img)
    mean = img.mean()
    std = img.std()
    img = (img - mean) / std 
    mean = img.mean()
    std = img.std()
    img = np.ndarray.tolist(img)
    return img

def TrainData(faces_path = faces_path_train, non_faces_path = non_faces_path_train):
    Training_Data = []
    
    i = 1
    for filename in os.listdir(faces_path):        
        if filename.endswith(".png"):
            if i > faces_size:
                break

            img=cv2.imread(faces_path+filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = normalize(img)
            data = (img,1)            
            Training_Data.append(data)
            i += 1
    faces_images_count = len(Training_Data)

    i = 1        
    for filename in os.listdir(non_faces_path):
        if filename.endswith(".png"):
            if i > non_faces_size:
                break
            img=cv2.imread(non_faces_path+filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = normalize(img)
            data = (img,0)
            Training_Data.append(data)
            i += 1
    non_faces_images_count = len(Training_Data)- faces_images_count
            
    random.shuffle(Training_Data)
    print('faces count : ',faces_images_count, ', non-faces count : ' , non_faces_images_count, ', Saved : models/trainingModel.pkl')

    Training = open('./models/trainingModel.pkl','wb')
    pickle.dump(Training_Data,Training)
    Training.close()

    NoOfClassifiers = 10
    window_size = 8
    classifier = FaceDetector(NoOfClassifiers)
    
    with open('./models/trainingModel.pkl', 'rb') as f:
        training = pickle.load(f)

    classifier.Train(training, faces_images_count, non_faces_images_count, window_size)
    classifier.save('./models/Classifier')


def TestData(faces_path = faces_path_test , non_faces_path = non_faces_path_test):
    Test_Data = []
    i = 1
    for filename in os.listdir(faces_path):
        if filename.endswith(".png"):
            if i > faces_size:
                break
            img=cv2.imread(faces_path+filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = normalize(img)
            data = (img,1)
            Test_Data.append(data)
            i += 1
    faces_images_count = len(Test_Data)

    i = 1        
    for filename in os.listdir(non_faces_path):
        if filename.endswith(".png"):
            if i > non_faces_size:
                break
            img=cv2.imread(non_faces_path+filename)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = normalize(img)
            data = (img,0)
            Test_Data.append(data)
            i += 1
    non_faces_images_count = len(Test_Data)- faces_images_count
            
    random.shuffle(Test_Data)
    print('faces count : ',faces_images_count, ', non-faces count : ' , non_faces_images_count, ', Saved : models/testingModel.pkl')
        
    Test = open('./models/testingModel.pkl','wb')
    pickle.dump(Test_Data,Test)
    Test.close()

    with open('./models/testingModel.pkl', 'rb') as f:
        Test = pickle.load(f)
    facedetector = FaceDetector()
    facedetector = facedetector.Load('./models/Classifier')
    i = len(facedetector.classifiers)
    fp = 0
    fn = 0
    accuracy = 0
    for image in Test:
        result = facedetector.classify(image[0])
        if result == image[1]:
            accuracy += 1
        if image[1] == 0 and result == 1:
            fp += 1
        if image[1] == 1 and result == 0:
                fn += 1

    print(facedetector.infos)
    print(facedetector.alphas)
    top_index = facedetector.alphas.index(max(facedetector.alphas))
    feature = facedetector.infos[top_index][0]
    top_threshold = facedetector.infos[top_index][1]
    training_accuracy, training_samples = facedetector.infos[top_index][2][0], facedetector.infos[top_index][2][1]
    print('Adaboost : ',i)
    print('Top Training Accuracy: {:.2%}'.format(training_accuracy/training_samples), '(', training_accuracy, '/',training_samples,')')
    print('Test Accuracy : {:.2%}'.format(accuracy/len(Test)), '(', accuracy, '/', len(Test),')')
    print('Test False Positive : {:.2%}'.
          format(fp/len(Test),), '(', fp, '/', len(Test),')')
    print('Test False Negative : {:.2%}'.format(fn/len(Test),), '(', fn, '/', len(Test),')')
TrainData() #uncomment this two lines before training and comment it after training is done
TestData()
