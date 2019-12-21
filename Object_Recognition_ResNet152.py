import numpy as np
import cv2
#from matplotlib import pyplot as plt
import fnmatch, os
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
#from keras.applications import ResNet50
import warnings
from ResNet152 import resnet152_model, SGD


#------------------------------------------------------------------------------
class ObjRec:

    #------------------------------------------------------
    def __init__(self, dir_name, model):
        self.dir_name = dir_name
        self.model = model
        self.incr = 0

    #------------------------------------------------------
    def listOfImages(self):
        listOfFiles = os.listdir(self.dir_name)
        #print(listOfFiles)
        pattern1 = "*.jpeg"
        pattern2 = "*.jpg"
        list_img=[]
        for entry in listOfFiles:
            if (fnmatch.fnmatch(entry, pattern1) or fnmatch.fnmatch(entry, pattern2)):
                list_img.append(self.dir_name + entry)
        return list_img

    #------------------------------------------------------
    def recognize(self, frame):
        try:
            '''
            frame = cv2.resize(frame,(224, 224),3)
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess(frame)
            '''
            im = cv2.resize(frame, (224, 224)).astype(np.float32)

            # Insert a new dimension for the batch_size
            im = np.expand_dims(im, axis=0)

            preds = self.model.predict(im)

            P = imagenet_utils.decode_predictions(preds)
            '''
            # loop over the predictions and display the rank-5 predictions +
            # probabilities to our terminal
            for (i, (imagenetID, label, prob)) in enumerate(P[0]):
                print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
            print()
            '''
            # load the image via OpenCV, draw the top prediction on the image,
            # and display the image to our screen
            (imagenetID, label, prob) = P[0][0]

            return (label, prob)
        except:
            return (None, None)

    #------------------------------------------------------
    def detect(self, img, type1):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # binarize the image
            ret, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # find connected components
            connectivity = 10
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
            #sizes = stats[1:, -1]; nb_components = nb_components - 1
            #min_size = 250 #threshhold value for objects in scene
            #img2 = np.zeros((img.shape), np.uint8)
            labels = []
            for i in range(0, nb_components+1):
                #print(img.shape, stats[i])
                # use if sizes[i] >= min_size: to identify your objects
                #color = np.random.randint(255,size=3)
                # draw the bounding rectangele around each object
                try:
                    if(stats[i][2] > 100 and stats[i][3] > 100):
                        img1 = img[stats[i][0]:stats[i][0]+stats[i][2], stats[i][1]:stats[i][1]+stats[i][3]]
                        (label, prob) = self.recognize(img1)
                        print('\n', label, prob)
                        if(label != None and label not in labels and prob > 0.47):
                            labels.append(label)
                            cv2.putText(img1, "Label: {}, {:.2f}%".format(label, prob * 100), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            img[stats[i][0]:stats[i][0]+stats[i][2], stats[i][1]:stats[i][1]+stats[i][3]] = img1
                            #cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
                            cv2.rectangle(img, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
                            #img2[output == i + 1] = color
                except:
                    continue

            if(type1 == 1):
                cv2.imwrite("./output/"+str(self.incr)+'.jpg', img)
                self.incr += 1
                cv2.imshow('ObjRec : ', img)
                key = cv2.waitKey(3000) #pauses for 3 seconds before fetching next image
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    __import__('sys').exit(1)
            else:
                #key = cv2.waitKey(2)
                return img
        except:
            return np.array([])
        return

#------------------------------------------------------------------------------
if __name__ == '__main__':

    #filter the warnings if any
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    selection = int(input('Enter 1 : Bulk of files, 2 : video : '))
    #face_csc = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    #eye_csc = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    preprocess = imagenet_utils.preprocess_input #ImageNet module for preprocessing input image.
    dir_name = input('Enter the path : ')
    #model = ResNet50(weights="imagenet")
    weights_path = 'resnet152_weights_tf.h5'
    # Test pretrained model
    model = resnet152_model(weights_path)
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    Obj = ObjRec(dir_name, model)

    if(selection == 1):
        list_of_files   =   Obj.listOfImages()

        for img in list_of_files:
            print(img)
            img1 = cv2.imread(img)

            Obj.detect(img1, 1)

        cv2.destroyAllWindows()

    else:
        try:
            file_name = dir_name.split('/')[-1].split('.')[0]
        except:
            file_name = None
        cap = cv2.VideoCapture(dir_name)
        count = 0
        writer = None
        while(cap.isOpened()):
            print('.', end = '')
            ret, frame = cap.read()
            count += 1
            if(count%5==0):
                img = Obj.detect(frame, 0)
                if len(img.shape) == 1:
                    break
                # check if the video writer is None
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter('./output/'+file_name+'.mp4', fourcc, 30, (img.shape[1], img.shape[0]), True)
                    # some information on processing single frame
                writer.write(img)

        writer.release()
        cap.release()
        cv2.destroyAllWindows()
