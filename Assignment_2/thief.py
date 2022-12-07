# -*- coding: utf-8 -*-
"""A2Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A-h8MCJPPSucckzDMGWggPjFygVwQ6k7
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image


def adjust_gamma(image,gamma=1.0):
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        image[x][y]=(image[x][y]/255)**(1/gamma)*255
  return image
  

def filter(image):
	g=2
	adjusted = adjust_gamma(image, gamma=g)
	# cv2.putText(adjusted, "g={}".format(g), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	# plt.subplot(1,2,2)
	adjusted=Image.fromarray(adjusted)
	plt.imshow(adjusted)
	# plt.axis("off")
	# plt.title("filtered")
	plt.savefig("img.png")
	adjusted.show()
	# plt.subplot(1,2,1)
	# plt.imshow(image)
	# plt.axis("off")
	# plt.title("original")

input_image=sys.argv[1]
image=Image.open(input_image)
img=np.array(image)

image= adjust_gamma(img, gamma=2)
adjusted=np.array(image)
smallest=np.amin(adjusted)
biggest=np.amax(adjusted)
for x in range(adjusted.shape[0]):
  for y in range(adjusted.shape[1]):
    adjusted[x][y]=(adjusted[x][y]-smallest)*(255/(biggest-smallest))

adjusted=Image.fromarray(adjusted)
plt.imshow(adjusted)
plt.savefig("img.png")
adjusted.show()