import numpy as np
import cv2
from skimage.filters import threshold_adaptive
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import label 
from skimage.measure import regionprops
from skimage.io import imread
from skimage.morphology import opening, closing
from skimage.morphology import square, diamond, disk 



broj = 0
prosli=[]

def nadjiCrtu(video):

    cap2 = cv2.VideoCapture(video) #Open video file
    
   
    
    cap2.set(1, 7)
    ret, frame = cap2.read();
    image=frame
    cap2.release() #release video file
    cv2.imwrite('image1.png', image)
    
    img=imread('image1.png')
    print img.shape
    #plt.imshow(img)
    img_gray = rgb2gray(img)
    #plt.imshow(img_gray, 'gray')
    #print img_gray.dtype
    #print img_gray
    
    img_tr = img_gray > 0.02  
    #plt.imshow(img_tr)
    #plt.imshow(img_tr, 'gray')
    
    img_tr_cl = opening(img_tr, selem=square(4))
    
    #plt.imshow(img_tr_cl, 'gray')

    labeled_img = label(img_tr_cl)  
    regions = regionprops(labeled_img)
    ##print "regionsi",  len(regions)

    
        

    regions_line = []
    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]  # visina
        w = bbox[3] - bbox[1]  # sirina
        if float(h) > 100 and w >150:
            regions_line.append(region)
            
            
    duzina =  len(regions_line[0].coords) -1       
    #plt.imshow(draw_regions(regions_barcode, img_barcode_tr_cl.shape), 'gray') 
    #print len(regions_line), regions_line[0].coords[0]
    #print len(regions_line), regions_line[0].coords[duzina]  
    pocetak = (regions_line[0].coords[duzina][1]-9, regions_line[0].coords[duzina][0]+9)
    kraj = (regions_line[0].coords[0][1]+9, regions_line[0].coords[0][0]-9)    
        
    #plt.imshow(draw_regions(regions_barcode, img_barcode_tr_cl.shape), 'gray')        
    #plt.imshow(draw_regions(regions_line, img_tr_cl.shape), 'gray')

    return pocetak, kraj



def draw_regions(regs, img_size):
    img_r = np.ndarray((img_size[0], img_size[1]), dtype='float32')
    #print "ovde sam"
    for reg in regs:
        coords = reg.coords  # coords vraca koordinate svih tacaka regiona
        for coord in coords:
            img_r[coord[0], coord[1]] = 1.
    return img_r

    


    
cv2.destroyAllWindows() #close all o