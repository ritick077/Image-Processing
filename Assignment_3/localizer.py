import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter

input_image=sys.argv[1]
image=Image.open(input_image)

map=np.array(image)
# print(map.shape)
# print(map.shape)
# road=map[142:256,78:140]
# grass=map[0:60,158:230]
# building=map[41:95,18:71]

# build=Image.fromarray(building)
# plt.imshow(build)
# build.show()

m_r=0
m_g=0
m_b=0
for i in range(map.shape[0]):
    for j in range(map.shape[1]):
        m_r+=map[i][j][0]
        m_g+=map[i][j][1]
        m_b+=map[i][j][2]

m_r/=(map.shape[0]*map.shape[1])
m_g/=(map.shape[0]*map.shape[1])
m_b/=(map.shape[0]*map.shape[1])


rg=round(abs(m_r-m_g),2)
rb=round(abs(m_r-m_b),2)
gb=round(abs(m_g-m_b),2)
gbr=round(abs(2*m_g-m_b-m_r),2)
# print(rg,rb,gb,gbr)

# location = [3.39,31.30,11.87]

# values=[abs(val-gb) for val in location]
# index = values.index(min(values))

# print(index+1)

# if index==0:
#     print("Building")
# elif index==1:
#     print("Grass")
# else:
#     print("Road")


if 0<=gb<=6 and 0<=gbr<=8 :
    print(1,"Building")
elif 0<=gb<=80 and 12<=gbr<=85 :
    print(2,"Grass")
elif 6<=gb<=20 and 0<=gbr<=12 :
    print(3,"Road")
else :
    print("Invalid", gb , gbr)