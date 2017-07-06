# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 00:05:37 2017

@author: ALex
"""

import sys
sys.path.append('code/')
import matplotlib.pyplot as plt
import training as p
import cv2
import numpy as np
from scipy import ndimage
from vector import  pnt2line, distance
from matplotlib.pyplot import cm 
import itertools
import time
import nalazenjeCrte as c
from skimage.color import rgb2gray

#p.train()  #trenira i save model u fajl 
p.readData() #cita istreniran model iz fajla model.h5
videos = ["video-0.avi", "video-1.avi", "video-2.avi", "video-3.avi", "video-4.avi", "video-5.avi", "video-6.avi", "video-7.avi", "video-8.avi", "video-9.avi"]
video = videos[5] #video koji se koristi za testiranje
pocetak, kraj = c.nadjiCrtu(video)

cap = cv2.VideoCapture(video)


line = [ pocetak, kraj ]
        
cc = -1
def nextId():
    global cc
    cc += 1
    return cc
    
def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal
    
#def sacuvajSliku(vreme):
#    cap2.set(1, vreme-10)
#    ret, img = cap2.read()
#    cv2.mwrite("slika%d.png" %counter, img)
# color filter
kernel = np.ones((3,3),np.uint8)
lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])

#boundaries = [
#    ([230, 230, 230], [255, 255, 255])
#]


elements = []
t =0
counter = 0
times = []

while(1):
    try:
        start_time = time.time()
        ret, img = cap.read()
       # print ret, img
        #(lower, upper) = boundaries[0]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(img, lower, upper)    
        img0 = 1.0*mask
    
        img0 = cv2.dilate(img0,kernel) #cv2.erode(img0,kernel)
        img0 = cv2.dilate(img0,kernel)
    
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled)
        for i in range(nr_objects):
            loc = objects[i]
            (xc,yc) = ((loc[1].stop + loc[1].start)/2,
                       (loc[0].stop + loc[0].start)/2)
            (dxc,dyc) = ((loc[1].stop - loc[1].start),
                       (loc[0].stop - loc[0].start))
    
            if(dxc>11 or dyc>11):
                cv2.circle(img, (xc,yc), 16, (25, 25, 255), 1)
                elem = {'center':(xc,yc), 'size':(dxc,dyc), 't':t}
                # find in range
                lst = inRange(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    elem['id'] = nextId()
                    elem['t'] = t
                    elem['pass'] = False
                    elem['history'] = [{'center':(xc,yc), 'size':(dxc,dyc), 't':t}]
                    elem['future'] = [] 
                    elements.append(elem)
                elif nn == 1:
                    lst[0]['center'] = elem['center']
                    lst[0]['t'] = t
                    lst[0]['history'].append({'center':(xc,yc), 'size':(dxc,dyc), 't':t}) 
                    lst[0]['future'] = [] 
                            
        for el in elements:
            tt = t - el['t']
            if(tt<3):
                dist, pnt, r = pnt2line(el['center'], line[0], line[1])
                if r>0:
                    cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                    c = (25, 25, 255)
                    if(dist<9):
                        c = (0, 255, 160)
                        if el['pass'] == False:
                            el['pass'] = True
                            el['passt'] = el['t']
                            
                            counter += 1
    
                
    
                id = el['id']
                
    
        elapsed_time = time.time() - start_time
        times.append(elapsed_time*1000)
        cv2.putText(img, 'Counter: '+str(counter), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(90,90,255),2)    
    
        #print nr_objects
        t += 1
        if t%10==0:
            print t
        #cv2.imshow('frame', img)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
        
        #out.write(img)
    except:
        print 'EOF'
        break
       
cap.release()



def deskew(img):
    SZ = 28
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.45*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

et = np.array(times)
print 'mean %.2f ms'%(np.mean(et))
cap2 = cv2.VideoCapture(video)
num =0
suma =0
for element in elements:
    if element['pass']:
        
        vreme = element['passt']-20
        for h in element['history']:
            if h['t']==vreme:
                centar = h['center']
            
       
       
        cap2.set(1, vreme)
        ret, img = cap2.read()
       
        
        crop_img = img[centar[1]-14:centar[1]+14, centar[0]-14:centar[0]+14, 0]
      
        #print crop_img.shape
              
        cv2.erode(crop_img,kernel)
        cv2.dilate(crop_img, kernel)
        cv2.erode(crop_img,kernel)
        
        cv2.dilate(crop_img, kernel)
        
             
        #crop_img = deskew(crop_img) #funkcija za ispravljanje rukopisa ukrivo
       
        
        #plt.imshow(crop_img, cmap="Greys")
        suma +=p.prepoznaj(crop_img)
               
        num+=1
        
        
cv2.destroyAllWindows()        
cap2.release()
print "suma za video ", video, " je ", suma 
