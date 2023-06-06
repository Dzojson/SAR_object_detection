import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu 



def road_segmentation(imgGrayRaw, imgRaw):
    #filtrowanie zdjęć
    imgFiltredB11 = cv2.bilateralFilter(imgGrayRaw, 30, 10, 10)
    imgFiltredB12 = cv2.bilateralFilter(imgFiltredB11, 10, 15,10)

    #progowanie 
    img_thresholded_inRange = cv2.inRange(imgFiltredB12, 100, 125)#145, 180
    _,img_thresholded = cv2.threshold(imgFiltredB12,  125, 255, cv2.THRESH_BINARY_INV)

    #pierwsze operacje mofrologiczne
    imgErode1 = cv2.morphologyEx(img_thresholded_inRange, cv2.MORPH_ERODE, np.ones((2,2),np.uint8), iterations=1)
    imgDilate1 = cv2.morphologyEx(imgErode1, cv2.MORPH_DILATE, np.ones((2,2),np.uint8), iterations=1)

    contours,_ = cv2.findContours(imgDilate1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    imageMask = np.zeros_like(imgDilate1)

    #usuwanie niechcianych konturów
    for countur in contours:
        area = cv2.contourArea(countur)
        rect = cv2.minAreaRect(countur)
        if area > 50:
                cv2.drawContours(imageMask, [countur], 0, 255, -1)

    binary_image_filtered = cv2.bitwise_and(imgDilate1, imgDilate1, mask=imageMask)

    #drugie zastosowanie operacji morfologicznych
    imgDilate2 = cv2.morphologyEx(binary_image_filtered, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations=1)
    imgErode2 = cv2.morphologyEx(imgDilate2, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations=2)
    imgDilate2 = cv2.morphologyEx(imgErode2, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations=1)

    contours,_ = cv2.findContours(imgDilate2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    imageMask = np.zeros_like(imgDilate1)

    #kolejne usuniecie "śmieciowych konturów"
    for countur in contours:
        area = cv2.contourArea(countur)
        rect = cv2.minAreaRect(countur)
        width, hight = rect[1]
        if area > 300 :
            cv2.drawContours(imageMask, [countur], 0, 255, -1)

    binary_image_filtered_2 = cv2.bitwise_and(imgDilate1, imgDilate1, mask=imageMask)

    contours,_ = cv2.findContours(binary_image_filtered_2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    imageMask = np.zeros_like(imageMask)

    #usunięcie konturów ewidętnie nie będących drogami
    for countur in contours:
        rect = cv2.minAreaRect(countur)
        width, hight = rect[1]

        if (width != 0 and hight != 0 ) and ((width / hight < 0.333 or width / hight > 3) or (hight/width < 0.333 or hight/width > 3)):
                cv2.drawContours(imageMask, [countur], 0, 255, -1)

    binary_image_filtered_wo_rect = cv2.bitwise_and(imgDilate1, imgDilate1, mask=imageMask)

    #trzecie zastosowanie operacji morfologicznych
    imgDilate4 = cv2.morphologyEx(binary_image_filtered_wo_rect, cv2.MORPH_DILATE, np.ones((5,5),np.uint8), iterations=3)
    contours,_ = cv2.findContours(imgDilate4, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    imageMask = np.zeros_like(imgDilate1)

    #ostania klasyfikacja konturó mająca być torami kolejowymi
    for countur in contours:
        rect = cv2.minAreaRect(countur)
        area = cv2.contourArea(countur)
        width, hight = rect[1]
        
        if ((width / hight < 0.5 or width / hight > 2) or (hight/width < 0.5 or hight/width > 2)) or area >= 1000:
            cv2.drawContours(imageMask, [countur], 0, 255, -1)
            
    road_mask = cv2.bitwise_and(imgDilate2, imgDilate2, mask=imageMask)


    return road_mask
