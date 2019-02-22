import os
import sys
import pandas as pd
data = pd.read_csv('/home/longino/Documents/DATA/Data_Entry_2017.csv')


datos = data.loc[data.FindingLabels.str.startswith('Mass') & data.FindingLabels.str.endswith('Mass')].ImageIndex
#datos = data.loc[data.FindingLabels.str.startswith('Nodule', na=False)].ImageIndex
#datos = datos.loc[data.FindingLabels.str.contains('Mass')].ImageIndex

image=datos.tolist()


source='/home/longino/Documents/DATA/imagenOriginal/images/'
destination='/home/longino/Documents/DATA/imagenOriginal/SoloMass/'
for img in image:
    os.rename(source+img, destination+img)
    
print("Dataset folders successfully created by breed name and copied all images in corresponding folders")
    
    