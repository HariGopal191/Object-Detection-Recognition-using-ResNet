
from ResNet152 import resnet152_model, SGD
from keras.preprocessing.image import ImageDataGenerator

model = resnet152_model()
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('tiny_imagenet/training',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('tiny_imagenet/validating',
                                            target_size = (224, 224),
                                            batch_size = 64,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
		            samples_per_epoch = 2230,
		            nb_epoch = 1,
		            validation_data = test_set,
		           	nb_val_samples = 298)

model.save("resnet152.h5")