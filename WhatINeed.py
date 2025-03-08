#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import matplotlib.pyplot as plt

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

#video=cv2.VideoCapture(args.image if args.image else 0)
#padding=20
#while cv2.waitKey(1)<0 :
#    hasFrame,frame=video.read()
#    if not hasFrame:
#        cv2.waitKey()
#        break

def predictGnA(face):
    global MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years')
    return (gender,age)

def MultiPersonPrediction(frame):
    global MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet

    padding = 20

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        gender,age = predictGnA(face)
        resultImg = cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    plt.figure()
    plt.imshow(resultImg)
    plt.title("Detecting age and gender")
    plt.show()

def SinglePersonPrediction(frame,vis=False):
    global MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet

    padding = 20

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if vis:resultImg = frame.copy()
    if not faceBoxes:
        print("No face detected")
        return None
    faceBox = faceBoxes[0]
    x1 = max(0,faceBox[0]-padding)
    x2 = min(faceBox[2]+padding, frame.shape[1]-1)
    y1 = max(0,faceBox[1]-padding)
    y2 = min(faceBox[3]+padding,frame.shape[0]-1)
    face=frame[y1:y2,x1:x2]

    gender,age = predictGnA(face)
    if vis:
        resutImg = cv2.rectangle(resultImg,(x1,y1),(x2,y2),(0,255,0),2)
        resultImg = cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        plt.figure()
        plt.imshow(resultImg)
        plt.title("Detecting age and gender")
        plt.show()
    return (gender,age)

if __name__ == "__main__":
    frame = cv2.imread("girl1.jpg")
    SinglePersonPrediction(frame,True)