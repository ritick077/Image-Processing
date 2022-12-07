# -*- coding: utf-8 -*-
"""EE604A_HA1_Q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KoNuKoUZcomjuiCcAn8OFINhiEWuNVeN
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# from google.colab.patches import cv2_imshow

def dotmatrix(K):
  output=np.zeros((300,500),dtype=np.uint8)
  w = 60
  r = 25
  # if K<10:
  #   K='0'+str(K)
  # else:
  #   K=str(K)
  if len(K)==1:
    K='0'+ K
  
  map = {}
  map['0']=[[1,1,1],
            [1,0,1],
            [1,0,1],
            [1,0,1],
            [1,1,1]]
  map['1']=[[0,1,0],
            [0,1,0],
            [0,1,0],
            [0,1,0],
            [0,1,0]]
  map['2']=[[1,1,1],
            [0,0,1],
            [1,1,1],
            [1,0,0],
            [1,1,1]]
  map['3']=[[1,1,1],
            [0,0,1],
            [1,1,1],
            [0,0,1],
            [1,1,1]]
  map['4']=[[1,0,1],
            [1,0,1],
            [1,1,1],
            [0,0,1],
            [0,0,1]]
  map['5']=[[1,1,1],
            [1,0,0],
            [1,1,1],
            [0,0,1],
            [1,1,1]]
  map['6']=[[1,1,1],
            [1,0,0],
            [1,1,1],
            [1,0,1],
            [1,1,1]]
  map['7']=[[1,1,1],
            [0,0,1],
            [0,0,1],
            [0,0,1],
            [0,0,1]]
  map['8']=[[1,1,1],
            [1,0,1],
            [1,1,1],
            [1,0,1],
            [1,1,1]]
  map['9']=[[1,1,1],
            [1,0,1],
            [1,1,1],
            [0,0,1],
            [1,1,1]]
  index=35
  for num in K:
    digit=map[num]
    for i in range(5):
      for j in range(3):
        if digit[i][j]:
          cent=(index+5+(10*j)+(2*j+1)*r,5+10*i+(2*i+1)*r)
          output=cv2.circle(output,cent,r,(255),-1)
    index+=180+w

  return np.uint8(output)

s=input()
image = dotmatrix(s)
# cv2_imshow(image)
plt.imshow(image)
pilimg = Image.fromarray(image)
pilimg.save("dotmatrix.jpg")
pilimg.show()