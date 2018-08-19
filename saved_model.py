import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import math
import sys
 
#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')￼
 
if __name__ == '__main__' :
 
    def getROI_Array(img):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
        (rect, weights) = hog.detectMultiScale(img, **hogParams)  
        return rect 
    
    sess = tf.Session() 

    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('saved_model-2.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    # Now, let's access and create placeholders variables
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name("Softmax:0")
    X = graph.get_tensor_by_name("X:0") 
    
    # Read video
    cap = cv2.VideoCapture("/home/jite/Downloads/ManU.mp4")
    
    counter = 0
    frameCount = 0
    rect = []
    roiList = []
    img = []
    c = 0
    temp = []   
    
    rect, img = cap.read()        
    
 
    # Exit if video not opened.
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = cap.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()

 
    while True:
        # Read a new frame
        ok, img = cap.read()
        if not ok:
            break
                 
        if(frameCount%2 == 0):
            rect = getROI_Array(img) 
            print("hog detection")
            del roiList[:]
            temp.append(rect)
            for (x, y, w, h) in rect:

                roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi, (28, 28))
                roiList.append(roi)
#                cv2.imwrite('/home/jite/Downloads/video2/video'+str(counter)+'.png',roi)

                images = []
                images.append(roi.ravel())
                y_test_images = np.zeros((1, 2)) 

                prediction = sess.run(output, feed_dict={X: images})  # run session ROI as feed
                prediction = prediction*100
#                print(output)

                if prediction[0][1] > 75.0:
                    cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),1)
                    font = cv2.FONT_HERSHEY_PLAIN
                    t = math.ceil(prediction[0][1])
                    cv2.putText(img,'Chelsea '+str(t)+"%",(x, y), font, 0.7,(200,200,255),1,cv2.LINE_AA)
#                    cv2.imwrite('/home/jite/Videos/ann/pos/'+str(counter)+'.png',roi)
                else:
                    cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),1)
                    font = cv2.FONT_HERSHEY_PLAIN
                    t = math.ceil(prediction[0][0])                
                    cv2.putText(img,'Manu '+str(t)+"%",(x, y ), font, 0.7,(200,200,255),1,cv2.LINE_AA)                    
#                    cv2.imwrite('/home/jite/Videos/ann/neg/'+str(counter)+'.png',roi)
            
        if len(roiList) > 3:
            if not type(rect) is bool:
                for r in rect:
                    t = tuple(r)
#                    print(t)
                    
        frameCount = frameCount + 1  
        print(frameCount)
        cv2.imshow('img',img)
        counter = counter + 1
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break