#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('img/test_background.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template_orig = cv2.imread('img/fish.png.', 0)
template = cv2.resize(template_orig, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
w, h = template.shape[::-1]
 
res = cv2.matchTemplate(template_orig, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res >= threshold)

img_res = img_rgb.copy()
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_res, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

titles = ['Original', 'Template Matching']
images = [img_rgb, img_res]

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))    
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:




