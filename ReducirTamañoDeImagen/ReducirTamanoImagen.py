import os
import sys
import pandas as pd
from PIL import Image

data = pd.read_csv('/home/longino/Documents/DATA/Masas/mass.csv')

datos = data.ImageIndex

image=datos.tolist()

filename = '/home/longino/Documents/prueba/masasVAL/'


for img in image:
	file_parts = os.path.splitext(filename+img)
	outfile = file_parts[0] + file_parts[1]
	size = 224, 224
	try:
		img = Image.open(filename+img)
		img = img.resize(size, Image.ANTIALIAS)
		img.save(outfile, "PNG")
	except IOError as e:
		print("An exception ocurred '%s'" % e)

print("Dataset folders successfully created by breed name and copied all images in corresponding folders")
