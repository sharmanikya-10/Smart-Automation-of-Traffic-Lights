# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:51:44 2022

@author: dell
"""

import cv2 
import matplotlib.pyplot as plt
import urllib
import time



# threshold parameter is a confidence level t dtect the object 
#

#setting the font type for the image 
font_scale= 1
font = cv2.FONT_HERSHEY_PLAIN



classNames = []
file_name ='level.txt'
with open(file_name ,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    classNames.append(f.read())
#print(classNames)
#print(len(classNames))   

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb' 

net = cv2.dnn_DetectionModel(frozen_model , config_file)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

#img = cv2.imread(r'C:\Users\dell\images.jpg')

def get_object(img , thres ,nms ,draw =True  , object = []):
    classIds , confs, bbox = net.detect(img, confThreshold = thres ,nmsThreshold = nms)
    print(classIds ,bbox)
    object_info = []
    if len(object)==0:
        object= classNames  
    
    if len(classIds) !=0:
          
        for classId ,confidence ,box in zip(classIds.flatten(),confs.flatten(), bbox):
            className = classNames[classId-1]
            if className in object :
                object_info.append([box,className])
                if(draw == True):
                    cv2.rectangle(img ,box,color =(0,255,0), thickness= 2)
                    cv2.putText(img, classNames[classId -1].upper() ,(box[0]+10 , box[1]+ 30),font, font_scale,(0,255,0),2)
        x =len(classIds)
        print(x)
        return img ,object_info ,x
        
    #if we want to create it as the module so the first thing we have to do is to given the name main so that it may start to run automatically
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(r'C:\Users\dell\Downloads\Car - 16849.mp4')
    
    cap.set(3,640)
    cap.set(4,480)
    cap.set(10,70)
    if not cap.isOpened():
        cap=  cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    
    while True:
        retr , frame = cap.read()
        frame = cv2.resize(frame, (500,500),interpolation = cv2.INTER_CUBIC)
        sam , object_info ,x = get_object(frame  ,thres =0.5 , nms =0.2  ,object =['car' ,'truck'])
        print(object_info)
        api = "https://api.thingspeak.com/update?api_key=54OGQ41N06J7FHDC&field1="
        #x = 20
        req = urllib.request.urlopen(api +str(x))
        print(x)
        cv2.imshow("Dip",frame)
        q =cv2.waitKey(1)
        if q== ord('q'):
           break
  
cap.release()
cv2.destroyAllWindows()       
        