import cv2
import numpy as np

#moduly zawierajace algorytmy zegmentacji poszczegolnych obiektow
from airport_river_segmentation import airport_river_segmentation
from rail_segmentation import rail_segmentation
from road_segmentation import road_segmentation

def main():
    img_path = 'D:\Studia\SEM7\praca_inz\zdjecia\ICEYE_QUICKLOOK_SLEA_208365_20211220T014333.png'

    #wczytanie obrazu
    imgRaw = cv2.imread(img_path)
    imgGrayRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2GRAY)

    #wywolanie funkcji segmentacji lotniska i rzeki
    airport_mask, river_mask = airport_river_segmentation(imgRaw, imgGrayRaw)
    #wywolanie fukcji segmentacji torow
    rail_mask = rail_segmentation(imgGrayRaw, imgRaw)

    #nalozenie masek na zdjecie SAR
    rail_color = np.array([255, 0, 0], dtype='uint8')
    masked_rail = np.where(rail_mask[...,None], rail_color, imgRaw)

    airport_color = np.array([0, 0, 255], dtype='uint8')
    masked_airport = np.where(airport_mask[...,None], airport_color, masked_rail)

    river_color = np.array([0, 255, 0], dtype='uint8')
    masked_river = np.where(river_mask[...,None], river_color, masked_airport)

    cv2.imwrite("D:\Studia\SEM7\praca_inz\images\SAR_Image_with_segmentation.png" , masked_river)

if __name__ == '__main__':
    main()
