import cv2
import numpy as np


def minmax(v):
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v


def dithering_gray(inMat, samplingF):
    #https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
    #input is supposed as color
    # grab the image dimensions
    h = inMat.shape[0]
    w = inMat.shape[1]

    # loop over the image
    for y in range(0, h-1):
        for x in range(1, w-1):
            # threshold the pixel
            old_p = inMat[y, x]
            new_p = np.round(samplingF * old_p/255.0) * (255/samplingF)
            inMat[y, x] = new_p

            quant_error_p = old_p - new_p

            # inMat[y, x+1] = minmax(inMat[y, x+1] + quant_error_p * 7 / 16.0)
            # inMat[y+1, x-1] = minmax(inMat[y+1, x-1] + quant_error_p * 3 / 16.0)
            # inMat[y+1, x] = minmax(inMat[y+1, x] + quant_error_p * 5 / 16.0)
            # inMat[y+1, x+1] = minmax(inMat[y+1, x+1] + quant_error_p * 1 / 16.0)

            inMat[y, x+1] = minmax(inMat[y, x+1] + quant_error_p * 7 / 16.0)
            inMat[y+1, x-1] = minmax(inMat[y+1, x-1] + quant_error_p * 3 / 16.0)
            inMat[y+1, x] = minmax(inMat[y+1, x] + quant_error_p * 5 / 16.0)
            inMat[y+1, x+1] = minmax(inMat[y+1, x+1] + quant_error_p * 1 / 16.0)


            #   quant_error  := oldpixel - newpixel
            #   pixel[x + 1][y    ] := pixel[x + 1][y    ] + quant_error * 7 / 16
            #   pixel[x - 1][y + 1] := pixel[x - 1][y + 1] + quant_error * 3 / 16
            #   pixel[x    ][y + 1] := pixel[x    ][y + 1] + quant_error * 5 / 16
            #   pixel[x + 1][y + 1] := pixel[x + 1][y + 1] + quant_error * 1 / 16


    # return the thresholded image
    return inMat

#read image
inMat = cv2.imread('cameraman.tif') #specify the image you want to use

#gray ditering
grayMat = cv2.cvtColor(inMat, cv2.COLOR_BGR2GRAY)
outMat_gray = dithering_gray(grayMat.copy(), 1)
cv2.imwrite('Dithered.png', outMat_gray)
cv2.imshow('Dithered',outMat_gray)
cv2.waitKey()
cv2.destroyAllWindows()
