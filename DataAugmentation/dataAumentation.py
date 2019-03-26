import os
import shutil
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class_dir = ['masas','nodulos']

for item in class_dir:
    
    # archivo temporal para almacenar imagenes
    tmp_dir = 'tmp_dir'
    if not os.path.exists(tmp_dir):
    	os.makedirs(tmp_dir)

    class_dir = os.path.join(tmp_dir, 'class_dir')

    if not os.path.exists(class_dir):
    	os.makedirs(class_dir)
   

    # list all images in that directory
    img_list = os.listdir(item)

    # Copy images from the class train dir to the img_dir e.g. class 'mel'
    for fname in img_list:
            # source path to image
            src = os.path.join(item, fname)
            # destination path to image
            dst = os.path.join(class_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)


    # point to a dir containing the images and not to the images themselves
    path = tmp_dir
    save_path = item

    # Create a data generator
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                           save_to_dir=save_path,
                                           save_format='png',
                                                    target_size=(224,224),
                                                    batch_size=batch_size)



    # Generate the augmented images and add them to the training folders
    
    ###########
    
    num_total_img = 10000 # total number of images we want to have in each class
    
    ###########
    
    num_files = len(class_dir)
    num_batches = int(np.ceil((num_total_img-num_files)/batch_size))
    for i in range(0,num_batches):
    	imgs, labels = next(aug_datagen)

    shutil.rmtree('tmp_dir')

