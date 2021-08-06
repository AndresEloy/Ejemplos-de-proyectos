import cv2
import numpy as np
import os 

for i in range(15):
    d = i + 1
    filename = "C:/Users/ANDY/Downloads/Actividad1_04/chequesJ/cheque%d.jpg"%d
    image = cv2.imread(filename)
    roi = image[380:525, 900:1300]
    img_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    t, dst = cv2.threshold(img_gris, 170, 255, cv2.THRESH_BINARY)
    filesaved = "C:/Users/ANDY/Downloads/Actividad1_04/Firmas/MyImage%d.jpg"%d
    cv2.imwrite(filesaved,dst)


