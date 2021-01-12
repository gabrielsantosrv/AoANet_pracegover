from PIL import Image
import pytesseract
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import re

def OCR (imageIn, texto):
    imagem = Image.open(imageIn).convert('RGB')
    npimagem = np.asarray(imagem).astype(np.uint8)  
    npimagem[:, :, 0] = 0 
    npimagem[:, :, 2] = 0
    img = cv.cvtColor(npimagem, cv.COLOR_RGB2GRAY) 
    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    images = [img, th1, th2, th3]

    a=pytesseract.image_to_string(th1,lang='por') 
    b=pytesseract.image_to_string(th2,lang='por')
    c=pytesseract.image_to_string(th3,lang='por')
    d=pytesseract.image_to_string(img,lang='por') 

    a=a.replace("\n", " ")
    b=b.replace("\n", " ")
    c=c.replace("\n", " ")
    d=d.replace("\n", " ")
    a=re.sub("\s\s+" , " ", a)
    b=re.sub("\s\s+" , " ", b)
    c=re.sub("\s\s+" , " ", c)
    d=re.sub("\s\s+" , " ", d)

    retorno=a

    if len(d) >= len(b) and len(d) >= len(c) and len(d) >= len(a):
        retorno=d
    elif len(a) > len(b) and len(a) > len(c) and len(a) > len(d):
        retorno=a
    elif len(b) > len(a) and len(b) > len(c) and len(b) > len(d):
        retorno=b
    elif len(c) > len(a) and len(c) > len(b) and len(c) > len(d):
        retorno=c
    else:
        retorno=d

    final=texto.replace("UNK", retorno)
    return(final) #retorna texto com UNK alterado

