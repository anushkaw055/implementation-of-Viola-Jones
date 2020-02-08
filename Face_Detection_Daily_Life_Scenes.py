import cv2
import numpy as np
import os
import pickle
from main import FaceDetector, Classifier

life_img_folder_path = './DailyLifeScenes/'

dirListing = os.listdir(life_img_folder_path)
print ('number of images in the folder : ' + str(len(dirListing)))

infile = open('./models/Classifier.pkl','rb')
clf = pickle.load(infile)
#clf = FaceDetector.Load('./models/Classifier.pkl')
def nonMaxSupAlgo(bounding_boxes, maxOverLap):
    if len(bounding_boxes) == 0: 
        return []
    pick = []
    x1 ,y1,x2, y2= bounding_boxes[:,0], bounding_boxes[:,1],bounding_boxes[:,2],bounding_boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(y2)

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)
        suppress = [last]
        
        for pos in range(0, last):
            j = indexes[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[j] 
            if overlap > maxOverLap:
                suppress.append(pos)

        indexes = np.delete(indexes, suppress)
        return bounding_boxes[pick]
def getFaceLocations(gray):
    print('Face detection started')
    rowssize = 50
    colssize = 50
    locations = []
    while rowssize<(len(gray)-2):
        for r in range(0,gray.shape[0] - rowssize, 10):
            for c in range(0,gray.shape[1] - colssize, 10):
                window = gray[r:r+rowssize,c:c+colssize]
                window=cv2.resize(window,dsize=(19,19))
                prediction = clf.classify(window)
                if prediction ==1:      
                    locations.append([r,c,r+rowssize,c+colssize])       
        colssize+=50
        rowssize+=50
    
    return locations
    
    
for i in range(1, len(dirListing)+1):       
    testimage = './DailyLifeScenes/testimage'+str(i)+'.jpg'

    imgtest = cv2.imread(testimage)                    
    gray = cv2.cvtColor(imgtest,cv2.COLOR_RGB2GRAY)
    print('Image loaded : ', testimage)

    mean = gray.mean() # Normalizing
    std = gray.std()
    std = gray.std()

    locations = getFaceLocations(gray)
            
    if locations:
        locations = nonMaxSupAlgo(np.array(locations), 0)
        print('Face(s) detected at the location : ', locations)
        for location in locations:
            cv2.rectangle(imgtest, (location[0], location[1]), (location[2], location[3]), (0,255,0), 5)
        cv2.imwrite("./DailyLifeScenes-result/testimage-result"+str(i)+".png",imgtest)
        print("Image is saved : ./DailyLifeScenes-result/testimage-result"+str(i)+".png")
        print('')
    else:
        print('No face detected')
