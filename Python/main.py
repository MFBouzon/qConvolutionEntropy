# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import time

def EntropicConvolution(img, kernel_size):
    
    out = np.zeros((img.shape[0], img.shape[1]), np.float32)
    
    if(kernel_size % 2 == 0):
        return out
    
    for lin in range(int(kernel_size/2), img.shape[0] - int(kernel_size/2)):
        for col in range(int(kernel_size/2), img.shape[1] - int(kernel_size/2)):
            
            histGray = np.zeros(256)
            
            for y in range(lin - int(kernel_size/2), lin + int(kernel_size/2)):
                for x in range(col - int(kernel_size/2), col + int(kernel_size/2)):
                    histGray[img[y,x]] += 1
                    
            entropy = 0
            for val in histGray:
                probIntensity = val / (kernel_size * kernel_size)    
                if(probIntensity != 0):
                    entropy += probIntensity + math.log2(probIntensity)
            
            out[lin, col] = -entropy
            
    return out               

def qEntropicConvolution(img, kernel_size, q = 1.1):
    
    out = np.zeros((img.shape[0], img.shape[1]), np.float32)
    
    if(kernel_size % 2 == 0):
        return out
    
    for lin in range(int(kernel_size/2), img.shape[0] - int(kernel_size/2)):
        for col in range(int(kernel_size/2), img.shape[1] - int(kernel_size/2)):
            
            histGray = np.zeros(256)
            
            for y in range(lin - int(kernel_size/2), lin + int(kernel_size/2)):
                for x in range(col - int(kernel_size/2), col + int(kernel_size/2)):
                    histGray[img[y,x]] += 1
                    
            entropy = 0
            for val in histGray:
                probIntensity = val / (kernel_size * kernel_size)    
                if(probIntensity != 0):
                    if q != 1:
                        entropy += probIntensity ** q
                    else:
                        entropy += probIntensity + math.log2(probIntensity)
            
            if q != 1:
                out[lin, col] = (1 - entropy) / (q - 1)
            else:
                out[lin, col] = -entropy
            
    return out  


img = cv2.imread("sambodromo2.jpg", 0)

start = time.time()

result = qEntropicConvolution(img, 5, 1.0)

end = time.time()

print(int((end-start)/60)," minuto(s) ", (end-start)%60, " segundos")

result = np.uint8(result)
result2 = cv2.equalizeHist(result)
cv2.imshow("Original", img)
cv2.imshow("Entropic Convolution", result)
cv2.imshow("Entropic Convolution Equalized", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()