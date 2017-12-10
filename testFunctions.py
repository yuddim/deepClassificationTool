
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
import time
import os
import numpy as np
from shutil import copyfile

def test_image(imagepath, img_width, img_height, model):
    img = load_img(imagepath, target_size=(img_width, img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x / 255
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    start = time.time()
    out = model.predict(x, batch_size=1)
    stop = time.time()
    sec = stop - start
    val2 = out.max()
    maxind = out.argmax()
    print('Image {0} Class = {1} Predict = {2} PredictTime = {3}'.format(imagepath, maxind, out, sec))

def test_two_image(imagepath1, imagepath2, img_width, img_height, model):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    x = np.zeros((2,)+input_shape, dtype=K.floatx())
    img = load_img(imagepath1, target_size=(img_width, img_height))
    x_1 = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x_1 = x_1 / 255
    x[0] = x_1
    img = load_img(imagepath2, target_size=(img_width, img_height))
    x_2 = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x_2 = x_2 / 255
    x[1] = x_2
    #x_1 = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    start = time.time()
    out = model.predict(x, batch_size=1)
    stop = time.time()
    sec = stop - start
    #val2 = out.max()
    #maxind = out.argmax()
    print('Predict = {0} PredictTime = {1}'.format(out, sec))

def test_image_gen(imagepath, img_width, img_height, model):
    img = load_img(imagepath, target_size=(img_width, img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    sample_datagen = ImageDataGenerator(rescale=1. / 255)
    sample_gen = sample_datagen.flow(x, batch_size=1)

    start = time.time()
    out = model.predict_generator(sample_gen, 1)
    stop = time.time()
    sec = stop - start
    val2 = out.max()
    maxind = out.argmax()
    print('Image {0} Class = {1} MaxPredict = {2} PredictTime = {3}'.format(imagepath, maxind, val2, sec))

def clearFolder(target_path):
    # delete all files in the directory
    filesToRemove = [f for f in os.listdir(target_path)]
    for f in filesToRemove:
        os.remove(os.path.join(target_path, f))

def test_path_gen(path, img_width, img_height, model, save_e_path = '', save_tune_path = '',save_errors=False, clear_tune_path=True):
    little_datagen = ImageDataGenerator(rescale=1. / 255)
    test_dir = path
    little_generator = little_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    start = time.time()
    out = model.predict_generator(little_generator, little_generator.n)
    #print(out)
    #print(little_generator.filenames)
    stop = time.time()
    sec = stop - start
    print("predict time = %.4f sec" % sec, end=' ')
    #print(out)

    #create result folder
    if(save_errors):
        path_list = list(little_generator.class_indices.keys())
        path_list.sort()
        os.makedirs(save_e_path, exist_ok=True)
        os.makedirs(save_tune_path, exist_ok=True)
        for folder in path_list:
            target_path = os.path.join(save_e_path,folder)
            os.makedirs(target_path, exist_ok=True)
            clearFolder(target_path)
            target_path2 = os.path.join(save_tune_path, folder)
            os.makedirs(target_path2, exist_ok=True)
            if clear_tune_path:
                clearFolder(target_path2)

    #calc precision and recall for each class

    precision = []
    recall = []
    n_classes = little_generator.num_class
    n_images = little_generator.n
    treshold = 0.5
    labels = little_generator.class_indices.items()
    print (labels)
    accuracy = 0
    all_facts = n_images
    real_facts = 0
    for i in range(n_classes):
        #calc precision and recall for i-th class. i starts from 0
        precision.append(0)
        recall.append(0)
        tp = 0
        fp = 0
        fn = 0
        for j in range(n_images):
            im_true_class = little_generator.classes[j]
            im_predict_result = out[j].max()
            im_predict_class = out[j].argmax()
            if (i == 0):
                #print('Image {0} Class = {1} MaxPredict = {2}'.format(j, im_predict_class, im_predict_result))
                if(im_true_class == im_predict_class):
                    real_facts += 1
                if(im_true_class != im_predict_class) and save_errors:
                    save_filename = os.path.basename(little_generator.filenames[j])
                    copyfile(os.path.join(path,little_generator.filenames[j]), os.path.join(save_e_path, path_list[im_predict_class], save_filename))
                    copyfile(os.path.join(path, little_generator.filenames[j]), os.path.join(save_tune_path, path_list[im_true_class], 'tune_'+save_filename))
            if (im_true_class == im_predict_class) and (im_predict_class == i):
                tp += 1
            else:
                if (im_predict_class == i) and (im_true_class != im_predict_class):
                    fp += 1
                else:
                    if (im_predict_class != i) and (im_true_class == i):
                        fn += 1
        if((tp + fp) != 0):
            precision[i] = tp / (tp + fp)
        if ((tp + fn) != 0):
            recall[i] = tp / (tp + fn)
    accuracy = real_facts / all_facts
    print('Accuracy {0}'.format(accuracy))
    print('Precision {0}'.format(precision))
    print('Recall {0}'.format(recall))
