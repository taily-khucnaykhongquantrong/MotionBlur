# MotionBlur
Motion Blur in python, code samples from https://github.com/lospooky/pyblur. This project adds customize kernel size capability comparing to the original one.    

## Example codes:
```python
import os
from cv2 import imread, imwrite

from MotionBlur import LinearMotionBlur


img = imread(imgPath)
# This should be an integer
blurIntensity = 15
# The direction to blur, can be any of number type
angle = 150
blurredImg = LinearMotionBlur(img, blurIntensity, angle, linetype="full")

imwrite(os.path.join(savePath, fileName.fileExt), resizedImg)
```
