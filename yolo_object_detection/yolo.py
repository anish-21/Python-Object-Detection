import cv2 as cv
import numpy as np

net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# print(classes)

cap = cv.VideoCapture('main_road.mp4')
# img = cv.imread('image.jpg')

while True:

    _, img = cap.read()
    height, width, _ = img.shape

    # prepare and convert 'img' to an input image which can be used in yolo. normalizes the image.
    blob = cv.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True) 

    # sets input from the blob into the network
    net.setInput(blob)

    # getUnconnectedOutLayersNames() function is used to get the output layer names
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = [] #bounding boxes
    confidences = []
    class_ids = [] #predicted classes

    for output in layerOutputs: #extracts all the information from layersOutput
        for detection in output: #extracts information from each of the output
            scores = detection[5:] # contains a list of all confidences of the 80 classes
            class_id = np.argmax(scores) # Returns the indices of the maximum values along an axis(only the highest value)
            confidence = scores[class_id] # Return the highest confidence in 'confidence' var.

            if confidence >= 0.5:
                centre_x = int(detection[0]*width) # scale normalized image back to original image, we multiply my *height, *width
                centre_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(centre_x - w/2)
                y = int(centre_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id) 

    # shows the number of boxes with their index locations. NMS --> non maximum supression 
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes.flatten())

    font = cv.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255, size=(len(boxes),3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
        
        


    cv.imshow('Image', img)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()    
cv.destroyAllWindows()