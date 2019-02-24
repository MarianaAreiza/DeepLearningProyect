import pandas as pd
import cv2
import os
import time
final=[]
drawing = False # true if mouse is pressed
ix,iy = -1,-1
def draw_rec(event,x,y,flags,param):
    global ix,iy,drawing,final

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),3)
        w=x-ix
        h=y-iy
        raw_data = {'ImageIndex': file,'x': ix,'y': iy,'w': w,'h': h}
        final.append(raw_data)
        print(raw_data)
path = '/home/longino/Documents/Nodulos/Nodule1/'
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            df=path+file
            img=cv2.imread(df)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',draw_rec)
            while(1):
                cv2.imshow('image',img)
                k = cv2.waitKey(1) & 0xFF
                if k == 13:
                    break
            cv2.destroyAllWindows()
df = pd.DataFrame(final, columns = ['ImageIndex', 'x', 'y', 'w', 'h'])
df.to_csv('example.csv')
print("Finalizado")
