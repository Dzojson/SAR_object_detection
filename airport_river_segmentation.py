import matplotlib.pyplot as plt
import numpy as np
import cv2


def airport_river_segmentation(imgRaw, imgGrayRaw):

    #filtrowanie obrazu
    imgFiltredB11 = cv2.bilateralFilter(imgGrayRaw, 30, 10, 10)
    imgFilterdM = cv2.medianBlur(imgFiltredB11, 9)

    #progowanie
    _, imgThresholded = cv2.threshold(imgFilterdM, 105, 255, cv2.THRESH_BINARY_INV) #105 jest idealnie do wykrywania rzeki i lotniska

    #operacje morfologiczne
    imgErode2 = cv2.morphologyEx(imgThresholded, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations=1)
    imgDilate2 = cv2.morphologyEx(imgErode2, cv2.MORPH_DILATE, np.ones((9,9),np.uint8), iterations= 2)

    contours,_ = cv2.findContours(imgDilate2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    imageMask = np.zeros_like(imgDilate2)

    #usuniecie niechcianych elemetnow
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)
        if area > 10000:
                cv2.drawContours(imageMask, [contour], 0, 255, -1)

    img_binary_filtered = cv2.bitwise_and(imgThresholded, imgThresholded, mask=imageMask)

    #wykrycie konturow
    contours,_ = cv2.findContours(img_binary_filtered, mode=cv2.RETR_EXTERNAL , method=cv2.CHAIN_APPROX_NONE)
    imageMask1 = np.zeros_like(imgDilate2)
    imageMask2 = np.zeros_like(imgDilate2)
    image = np.zeros_like(imgRaw)

    #klasyfikacja obiektow
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)
        
        width, hight = rect[1]
        if area > 100000 and area < 1000000:
            cv2.drawContours(imageMask1, [contour], 0, 255, -1)

        elif  (width != 0 and hight != 0 ) and (min(width/hight, hight/width) < 0.3) and area > 2000:
            cv2.drawContours(imageMask2, [contour], 0, 255, -1)

    river_mask_early = cv2.bitwise_and(imgThresholded, imgThresholded, mask=imageMask2)
    
    #wykrycie konturow
    contours,_ = cv2.findContours(river_mask_early, mode=cv2.RETR_EXTERNAL , method=cv2.CHAIN_APPROX_NONE)
    imageMask2 = np.zeros_like(imgDilate2)

    #sprawdzanie odleglosci konturow
    for contour in contours:
        for contour_tmp in contours:
            if contour[0][0][0] != contour_tmp[0][0][0]:      
                for cntP_tmp in contour_tmp:
                    x, y = cntP_tmp[0]
                    pt = (float(x),float(y))
                    distance = cv2.pointPolygonTest(contour, pt, True)
                    if abs(distance)  < 500:
                        cv2.drawContours(imageMask2, [contour], 0, 255, -1)

    airport_mask = cv2.bitwise_and(imgDilate2, imgDilate2, mask=imageMask1)

    river_mask = cv2.bitwise_and(imgDilate2, imgDilate2, mask=imageMask2)

    return airport_mask, river_mask
