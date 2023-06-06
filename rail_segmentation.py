import matplotlib.pyplot as plt
import numpy as np
import cv2


def rail_segmentation(imgGrayRaw, imgRaw):
    #filtrowanie zdjęć
    imgFiltredB11 = cv2.bilateralFilter(imgGrayRaw, 30, 10, 10)
    imgFiltredB12 = cv2.bilateralFilter(imgFiltredB11, 10, 15,10)

    #progowanie 
    img_thresholded_inRange = cv2.inRange(imgFiltredB12, 145, 180)#145, 180

    #pierwsze operacje mofrologiczne
    imgErode1 = cv2.morphologyEx(img_thresholded_inRange, cv2.MORPH_ERODE, np.ones((2,2),np.uint8), iterations=1)
    imgDilate1 = cv2.morphologyEx(imgErode1, cv2.MORPH_DILATE, np.ones((2,2),np.uint8), iterations=1)

    contours,_ = cv2.findContours(imgDilate1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    imageMask = np.zeros_like(imgDilate1)

    #usuwanie "śmieciowych" konturów
    for countur in contours:
        area = cv2.contourArea(countur)
        rect = cv2.minAreaRect(countur)
        if area > 10:
                cv2.drawContours(imageMask, [countur], 0, 255, -1)

    binary_image_filtered = cv2.bitwise_and(imgDilate1, imgDilate1, mask=imageMask)

    #drugie zastosowanie operacji morfologicznych
    imgDilate2 = cv2.morphologyEx(binary_image_filtered, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations=3)
    imgErode2 = cv2.morphologyEx(imgDilate2, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations=2)
    imgDilate2 = cv2.morphologyEx(imgErode2, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations=1)

    contours,_ = cv2.findContours(imgDilate2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    imageMask = np.zeros_like(imgDilate1)

    #kolejne usuniecie "śmieciowych konturó"
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

        if (width != 0 and hight != 0 ) and ((min(width / hight,hight / width) < 0.333)):
                cv2.drawContours(imageMask, [countur], 0, 255, -1)

    binary_image_filtered_wo_rect = cv2.bitwise_and(imgDilate1, imgDilate1, mask=imageMask)

    #trzecie zastosowanie operacji morfologicznych
    imgDilate4 = cv2.morphologyEx(binary_image_filtered_wo_rect, cv2.MORPH_DILATE, np.ones((5,5),np.uint8), iterations=3)
    contours,_ = cv2.findContours(imgDilate4, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    imageMask = np.zeros_like(imgDilate1)

    #ostania klasyfikacja konturow mająca być torami kolejowymi
    for countur in contours:
        rect = cv2.minAreaRect(countur)
        area = cv2.contourArea(countur)
        width, hight = rect[1]
        
        if min(width / hight,hight / width) < 0.5 or area >= 1000:
            cv2.drawContours(imageMask, [countur], 0, 255, -1)
            
    image_b4_connection = cv2.bitwise_and(imgDilate4, imgDilate4, mask=imageMask)

    contours,_ = cv2.findContours(image_b4_connection, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    imageMask = np.zeros_like(imgDilate1)
    image_with_connetion = image_b4_connection.copy()

    #laczenie segmentow torow
    for contour in contours:
        minDistance = None
        for contour_tmp in contours:
            if contour[0][0][0] != contour_tmp[0][0][0]:     
                x_1, y_1 = None, None 
                for cntP_tmp in contour_tmp:
                    x_tmp, y_tmp = cntP_tmp[0]
                    pt_tmp = (float(x_tmp),float(y_tmp))
                    distance_tmp = abs(cv2.pointPolygonTest(contour, pt_tmp, True))
                    if minDistance is None:
                        minDistance = distance_tmp
                        x_1, y_1 = x_tmp, y_tmp
                    elif minDistance > distance_tmp:
                        minDistance = distance_tmp
                        x_1, y_1 = x_tmp, y_tmp

                if abs(minDistance)  < 150:
                    for cntP in contour:
                        x, y = cntP[0]
                        pt = (float(x),float(y))
                        distance = abs(cv2.pointPolygonTest(contour_tmp, pt, True))
                        if distance == minDistance:
                            cv2.line(image_with_connetion, (x,y), (x_1, y_1), 255, thickness = 15 )
                            break

    
    return image_with_connetion

