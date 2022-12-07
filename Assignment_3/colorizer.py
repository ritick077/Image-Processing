import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image,ImageFilter

input_1=sys.argv[1]
Y = Image.open(input_1)
Y1=np.array(Y)

input_2=sys.argv[2]
cb = Image.open(input_2)
cb1=np.array(cb)

# print(cb1.shape)

input_3=sys.argv[3]
cr = Image.open(input_3)
cr1=np.array(cr)

for i in range(2):
    cb1=cv2.pyrUp(cb1)
    cr1=cv2.pyrUp(cr1)

cb1=cb1[1:623,0:960]
cr1=cr1[1:623,0:960]

# print(Y1.shape,cb1.shape,cr1.shape)



# img=Image.fromarray(ycbcr)
# plt.imshow(img)
# img.show()
# print(ycbcr.shape)
# rgb=np.zeros(ycbcr.shape)
# print("hello",rgb.shape)


# ycbcr=cv2.merge([Y1,cb1,cr1])
    

# values_matrix = np.array([[1, 1, 1], 
#                         [0, -0.344136, 1.772], 
#                         [1.402, -.714136, 0]])

Y1=Y1.astype(float)
cb1=cb1.astype(float)-128
cr1=cr1.astype(float)-128

r=Y1+1.402*cr1
g=Y1-0.344136*cb1-0.714136*cr1
b=Y1+1.772*cb1

b=np.clip(b,0,255)
g=np.clip(g,0,255)
r=np.clip(r,0,255)

rgb=cv2.merge([r,g,b])

# print(rgb)

rgb_img=Image.fromarray(np.uint8(rgb))
plt.imshow(rgb_img)
rgb_img.show()



