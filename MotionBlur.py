import math

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import line

from LineDictionary import LineDictionary

lineTypes = ["full", "right", "left"]


def LinearMotionBlur_random(img):
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = np.random.randint(0, 20)
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)

    return LinearMotionBlur(img, lineLength, lineAngle, lineType)


def LinearMotionBlur(imgarray, dim, angle, linetype):
    """Return an motion blurred image in ndarray type
    
    Parameters:
    -------
    imgarray : ndarray
        Input image.\\
    dim : int
        Size of kernel {3, 5, 7, 9}.\\
    angle : int
        Which direction the image is blurred.\\
    linetype : str
        Controls whether the blur kernel will be applied in full or only the
        left/right halves of it {'left', 'right', 'full'}.\\

    Returns:
    -------
    ndarray
        Returning value.
    """
    kernel = LineKernel(dim, angle, linetype)
    if imgarray.ndim == 3 and imgarray.shape[-1] == 3:
        convolved = np.stack(
            [
                convolve2d(
                    imgarray[..., channel_id], kernel, mode="same", fillvalue=255.0
                ).astype("uint8")
                for channel_id in range(3)
            ],
            axis=2,
        )
    else:
        convolved = convolve2d(imgarray, kernel, mode="same", fillvalue=255.0).astype(
            "uint8"
        )
    return convolved


def LineKernel(dim, angle, linetype="full"):
    kernelwidth = dim
    kernelCenter = int(math.floor(dim / 2))
    angle = SanitizeAngleValue(kernelCenter, angle)
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    lineDict = LineDictionary(kernelwidth)
    lineAnchors = lineDict.lines[dim][angle]

    if linetype == "right":
        lineAnchors[0] = kernelCenter
        lineAnchors[1] = kernelCenter
    if linetype == "left":
        lineAnchors[2] = kernelCenter
        lineAnchors[3] = kernelCenter

    rr, cc = line(lineAnchors[0], lineAnchors[1], lineAnchors[2], lineAnchors[3])
    kernel[rr, cc] = 1
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def SanitizeAngleValue(kernelCenter, angle):
    numDistinctLines = kernelCenter * 4
    angle = math.fmod(angle, 180.0)
    validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
    angle = nearestValue(angle, validLineAngles)
    return angle


def nearestValue(theta, validAngles):
    idx = (np.abs(validAngles - theta)).argmin()
    return validAngles[idx]


def randomAngle(kerneldim):
    kernelCenter = int(math.floor(kerneldim / 2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])
