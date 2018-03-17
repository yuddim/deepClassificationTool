import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import optimizers
import time
from modelInceptionV3m import inceptionV3m, inceptionV3Only
from modelResNetM import resNetM
from modelVGGm import VGGm
from testFunctions import test_image, test_two_image, test_image_gen, test_path_gen

# dimensions of our images.
img_width, img_height = 135, 76

train_data_dir = 'D:/Datasets/copters/train'
validation_data_dir = 'D:/Datasets/copters/test'

train_tune_data_dir = 'D:/Datasets/copters/train'
validation_tune_data_dir = 'D:/Datasets/copters/test'

epochs = 30
tune_epochs = 10
batch_size = 16
train_flag = True
tune_flag = False
#weights_path = 'car_5angles_weights_50_30_inceptionv3_wtd.h5'#'cars_direct_1.h5'
weights_path = 'camera_on_off_VGG.h5'
weights_tuned_path = 'camera_on_off_tuned.h5'#'cars_direct_tuned.h5'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#model = inceptionV3m(input_shape, 5)
#
#model = inceptionV3Only(input_shape, 5)
model = VGGm(input_shape, 5)
#model = resNetM(input_shape, 5)


#adam = optimizers.Adam(lr=1e-5)
sgd = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=True)

if(train_flag == True):
    if(tune_flag):
        model.load_weights(weights_path)

    # Training of neural network
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, #adam, #'sgd',
                  metrics=['accuracy'])
    print('Model is compiled\n')
    model.summary()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    if not tune_flag:
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    else:
        train_generator = train_datagen.flow_from_directory(
            train_tune_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_tune_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    print('Data is generated from folders\n')

    epochs_train = epochs
    if tune_flag:
        epochs_train = tune_epochs

    start = time.time()

    nb_train_samples = train_generator.samples
    nb_validation_samples = validation_generator.samples

    callbacks = [keras.callbacks.ModelCheckpoint(
        'D:/Projects/deepClassificationTool-master/models/camera_on_off_weights.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5',
        verbose=1,
        save_weights_only=True)]

    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_train,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    stop = time.time()
    sec = stop - start
    print("ConvNet is trained! Training time = %.4f sec" % sec, end=' ')
    print(hist.history)
    if(tune_flag):
        model.save_weights(weights_tuned_path)
    else:
        model.save_weights(weights_path)
    print('ConvNet is saved\n')
    K.clear_session()
else:
    #Testing of neural network
    if (tune_flag):
        model.load_weights(weights_tuned_path)
    else:
        model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, #adam, #'sgd'
                  metrics=['accuracy'])
    print('Model is compiled\n')


    test_path_gen('D:/Datasets/copters/train', img_width, img_height, model,
                  save_e_path='D:/Datasets/copters/errorresults_special_train_vgg',
                  save_tune_path='D:/Datasets/copters/tune_special_train_vgg', save_errors=True,
                  clear_tune_path=False)
    test_path_gen('D:/Datasets/copters/test', img_width, img_height, model,
                  save_e_path='D:/Datasets/copters/errorresults_special_vgg',
                  save_tune_path='D:/Datasets/copters/tune_special_vgg', save_errors=True, clear_tune_path=False)
    """
    test_path_gen('D:/Datasets/cars_angles/validation', img_width, img_height, model,
                  save_e_path='D:/Datasets/cars_angles/errorresults_special_validation_resnet',
                  save_tune_path='D:/Datasets/cars_angles/tune_special_validation_resnet', save_errors=False,
                  clear_tune_path=False)
    """
    """
    test_image('D:/Datasets/cars_angles/special_test/1_1 (2).JPG', img_width, img_height, model)
    test_image('D:/Datasets/cars_angles/special_test/1_2 (2).JPG', img_width, img_height, model)
    test_two_image('D:/Datasets/cars_direction/special_test/back/1_0 (2).JPG', 'D:/Datasets/cars_direction/special_test/back/1_3 (2).JPG', img_width, img_height, model)
    """
    K.clear_session()