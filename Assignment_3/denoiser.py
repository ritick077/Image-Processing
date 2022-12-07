import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter

input_image=sys.argv[1]
img=Image.open(input_image)
img2=np.array(img)

def gaussian(x,y,sigma):
  return (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*(sigma**2)))

def gr(size,sigma):
  new = np.zeros([size,size])
  sum=0
  for i in range ( -int((size-1)/2),int((size+1)/2)):
    for j in range ( -int((size-1)/2),int((size+1)/2)):
      x=int(i+(size-1)/2)
      y=int(j+(size-1)/2)
      new[x][y]=gaussian(i,j,sigma)
      sum+=new[x][y]
  # if sum!=0 :
  #   new1=np.around(new*16)
  #   print(new1)
  return new

def gi(Ip,Iq):
  sr=35
  return (1/(2*np.pi*(sr**2)))*np.exp(-(abs(Ip-Iq)**2)/(2*(sr**2)))

def convolve(x1,y1,img):
  k = int((size-1)/2) 
  sum=0
  for i in range(-k,k+1):
    for j in range(-k,k+1):         
      if(x1+i<0):
        continue
      if(y1+j<0):
        continue
      if(x1+i>=img.shape[0]):
        continue
      if(y1+j>=img.shape[1]):
        continue
      # print(sum)
      kr=gi(img[x1][y1],img[x1+i][y1+j])  
      # print(kr)
      sum+=img[x1+i][y1+j]*kernel[i+k][j+k]
  return np.round(sum)

def gauss(img,size,sigma):
  new=np.zeros(img.shape)
  X=img.shape[0]
  Y=img.shape[1]
  # print(X,Y)
  for i in range(X):
    for j in range(Y):
      new[i][j]=convolve(i,j,img)
  return new

X=img2.shape[0]
Y=img2.shape[1]
# print(X,Y)
size=1
sigma=0.8
kernel=gr(size,sigma)
img1= cv2.bilateralFilter(img2,20,45,65)
b,g,r = cv2.split(img1)
#print(kernel) 
newb=gauss(b,size,sigma)
newg=gauss(g,size,sigma)
newr=gauss(r,size,sigma)
newr=np.round((newr/newr.max())*255)
newb=np.round((newb/newb.max()) *255)
newg=np.round((newg/newg.max()) *255)
bgr=cv2.merge([newb,newg,newr])

# output=Image.fromarray(bgr)

bgr1=Image.fromarray(np.uint8(bgr))
plt.imshow(bgr1)
bgr1.show()