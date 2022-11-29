import os
from PIL import Image, ImageOps
import numpy as np

imagefolder = "/trainingimages/"
for file in os.listdir(os.getcwd()+imagefolder):
	if file.endswith(".jpeg"):
		img = Image.open(os.getcwd()+imagefolder+file)
		print(type(img))
        #img = ImageOps.grayscale(img)
	print(np.array(img).shape)
	if np.array(img).shape != (400,400):
		img.crop((0, 0, 400, 400)).save(os.getcwd()+imagefolder+file)
	elif np.array(img).shape == (400,400):
		img = (float(img)/np.max(img))*255
		print(img[0])
