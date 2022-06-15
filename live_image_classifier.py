import sys
import logging
import os
import cv2
import numpy as np
from utils import write_image, key_action, init_cam
from tensorflow.keras.models import load_model

# load trained image_classifier model
model = load_model("model_imageclassifier.h5")

# define dictionary (build for model) - transforms numerical prediction to category_name
dict_classes = {0: 'book',
                1: 'cutlery',
                2: 'face',
                3: 'gesture',
                4: 'glass',
                5: 'mug',
                6: 'pen',
                7: 'shoe'}

# fuction - takes captured image, applies transformations used in model, makes prediction
def predict_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.array(image)
    image_scaled = image.astype("float32")/255
    prediction = np.argmax(model.predict(image_scaled.reshape(-1, 224, 224, 3)))
    pred_class = dict_classes[prediction]
    print (f'I see a {pred_class}')
    return(pred_class)



if __name__ == "__main__":

    # folder to write images to
    out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
            
            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[y:y+width, x:x+width, :]
                #write_image(out_folder, image) 
                text = predict_frame(image)
                cv2.putText(frame, text, (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            

    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
