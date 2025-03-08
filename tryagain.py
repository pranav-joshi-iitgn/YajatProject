import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import numpy as np
from ultralytics import YOLO, solutions
import supervision as sv
from inference import get_model
import math

#A Gender and Age Detection program by Mahesh Sawant

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

def predictGnA(face):
    global MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    #print(f'Gender: {gender}')
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    #print(f'Age: {age[1:-1]} years')
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
	if not faceBoxes:return None
	faceBox = faceBoxes[0]
	x1 = max(0,faceBox[0]-padding)
	x2 = min(faceBox[2]+padding, frame.shape[1]-1)
	y1 = max(0,faceBox[1]-padding)
	y2 = min(faceBox[3]+padding,frame.shape[0]-1)
	if (x2-x1 < 10) or (y2-y1 < 10) :return None
	face=frame[y1:y2,x1:x2]
	gender,age = predictGnA(face)
	#except:
	#	plt.figure()
	#	plt.imshow(face)
	#	plt.show()
	if vis:
	    resutImg = cv2.rectangle(resultImg,(x1,y1),(x2,y2),(0,255,0),2)
	    resultImg = cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
	    plt.figure()
	    plt.imshow(resultImg)
	    plt.title("Detecting age and gender")
	    plt.show()
	return ((x1,y1),(x2,y2),gender,age)

## Another face detection model

from myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

PATH_TO_CKPT_HEAD = 'models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'
head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)
detector = YOLO("yolov10n.pt")


## Ultralytics supervision

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

video_info = None
byte_track = None
frame_generator = None
polygon_zone = None
view_transformer = None

SOURCE = np.array([[731,176], [1078,182], [1432,877], [449,862]])
TARGET_WIDTH = 6.5
TARGET_HEIGHT = 18.5
TARGET = np.array(
		[
		    [0, 0],
		    [TARGET_WIDTH - 1, 0],
		    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
		    [0, TARGET_HEIGHT - 1],
		]
	)
	
confidence_threshold=0.3
iou_threshold=0.5

def setVideo(
	VIDEO_SRC = "videos/mall.mp4",
	TARGET_WIDTH = 6.5,
	TARGET_HEIGHT = 18.5,
):
	global video_info,byte_track,frame_generator,polygon_zone,view_transformer,confidence_threshold,iou_threshold,SOURCE,TARGET
	video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_SRC)
	byte_track = sv.ByteTrack(
    	frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
	)

	thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
	text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)*0.5
	box_annotator = sv.BoxAnnotator(thickness=thickness)
	label_annotator = sv.LabelAnnotator(
		text_scale=text_scale,
		text_thickness= thickness//2,
		text_position=sv.Position.TOP_CENTER,
	)
	trace_annotator = sv.TraceAnnotator(
		thickness=thickness,
		trace_length=video_info.fps * 2,
		position=sv.Position.TOP_CENTER,
	)

	frame_generator = sv.get_video_frames_generator(source_path=VIDEO_SRC)
	#polygon_zone = sv.PolygonZone(polygon=SOURCE)
	view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


def detect(I):
	global detector,video_info,byte_track,confidence_threshold,iou_threshold
	result = detector(I)[0]
	detections = sv.Detections.from_ultralytics(result)
	detections = detections[detections.class_id == 0]
	detections = detections[detections.confidence > confidence_threshold]
	#detections = detections[polygon_zone.trigger(detections)]
	detections = detections.with_nms(threshold=iou_threshold)
	#byte_track = sv.ByteTrack(
    #	frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
	#)
	detections = byte_track.update_with_detections(detections=detections)
	#points = detections.get_anchors_coordinates(
	#            anchor=sv.Position.BOTTOM_CENTER
	#        )
	return detections
	
def head(I,p1,p2,vis=False,confidence_threshold=0.3):
	x1,y1 = p1
	x2,y2 = p2
	image = I[y1:y2,x1:x2]
	im_height, im_width, im_channel = image.shape
	anim, heads = head_detector.run(image, im_width, im_height)
	heads = [h for h in heads if h['confidence'] > confidence_threshold]
	if len(heads) == 0 : return None
	# gaussian probability with width/2 as sd
	def P(h):
		X = min(abs(x2-x1),abs(y2-y1))
		prior = np.exp(-2*(
			h['left']**2 + (X-h['right'])**2 +
			h['top']**2 + (X-h['bottom'])**2
		)/X**2)
		posterior =prior *h['confidence']
		return posterior
	heads = sorted(heads,key=P,reverse=True)
	h = heads[0]
	x1,x2 = h['left'],h['right']
	y1,y2 = h['top'],h['bottom']
	if (x2-x1) < 10 or (y2-y1) < 10:return None
	gender,age = predictGnA(h['cropped'])
	#if vis:
		#I = cv2.rectangle(I,(x1,y1),(x2,y2),(255,0,0),2)
		#plt.figure()
		#plt.imshow(h['cropped'])
	return ((x1,y1),(x2,y2),gender,age)
	
def head2(I,p1,p2,vis=False):
	x1,y1 = p1
	x2,y2 = p2
	person = I[y1:y2,x1:x2]
	return SinglePersonPrediction(person)

def headtest():
	#imgsrc = "sample.jpg"
	#I = cv2.imread(imgsrc)
	setVideo()
	for i in range(500):next(frame_generator)
	I = next(frame_generator)
	plt.imshow(I)
	plt.show()
	image = I.copy()
	im_height, im_width, im_channel = image.shape
	anim, heads = head_detector.run(image, im_width, im_height)
	#print(heads[0].keys())
	heads = [h for h in heads if h['confidence'] > 0.92]
	#print(heads)
	for h in heads:
		x1,x2 = h['left'],h['right']
		y1,y2 = h['top'],h['bottom']
		I = cv2.rectangle(I,(x1,y1),(x2,y2),(0,255,0),2)
	plt.imshow(I)
	plt.show()
	
def fulldetect(I,vis=False,Head_Detector = head2):
	image = I
	if vis:image = I.copy()
	out = detect(image)
	points = out.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
	points = view_transformer.transform_points(points=points).astype(int)
	D = {}
	for tid,cid,xyxy,p in zip(out.tracker_id,out.class_id,out.xyxy,points):
		if cid != 0:continue
		x1,y1,x2,y2 = xyxy.astype(int)
		X1,Y1 = p
		if vis:
			image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
			image = cv2.putText(image,f"tid : {tid}",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
		h = Head_Detector(I,(x1,y1),(x2,y2))
		#print(h)
		if h is None:
			D[tid] = [(x1,y1),(x2,y2),(X1,Y1),None,None,(None,None)]
			continue
		p1,p2,gender,age = h
		hx1,hy1 = p1
		hx2,hy2 = p2
		hx1 += x1
		hy1 += y1
		hx2 += x1
		hy2 += y1
		D[tid] = [(x1,y1),(x2,y2),(X1,Y1),(hx1,hy1),(hx2,hy2),(gender,age)]
		if vis:
			if gender is None:gender =""
			if age is None: age =""
			image = cv2.rectangle(image,(hx1,hy1),(hx2,hy2),(255,0,0),2)
			image = cv2.putText(image,f"{age} {gender}",(hx1,hy1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
	#print(out.xyxy)
	if not vis:return D
	plt.figure()
	plt.imshow(image)
	plt.show()
	return D
	
def markVideo(src="videos/MVI_0152.mp4",target_video_path ="out.avi",k=5):
	setVideo(src)
	coordinates = defaultdict(lambda: deque(maxlen=k*video_info.fps))
	Gender = defaultdict(lambda: deque(maxlen=k*video_info.fps))
	Age = defaultdict(lambda: deque(maxlen=k*video_info.fps))
	with sv.VideoSink(target_video_path, video_info) as sink:
		frames_done = 0
		for frame in frame_generator:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			D = fulldetect(frame)
			for tid in D:
				coordinates[tid].append(D[tid][2])
				g,a = D[tid][5]
				Gender[tid].append(g)
				Age[tid].append(a)
			for tracker_id in D:
				if len(coordinates[tracker_id]) < video_info.fps / 2:
					D[tracker_id].append(None)
				else:
					x1,y1 =coordinate_start = coordinates[tracker_id][-1]
					x0,y0 = coordinate_end = coordinates[tracker_id][0]
					#print("start :",coordinate_start)
					#distance = abs(coordinate_start - coordinate_end)
					distance = (x1 - x0)**2 + (y1 - y0)**2
					distance = distance**0.5
					time = len(coordinates[tracker_id]) / video_info.fps
					speed = distance / time
					speed = speed * 3.6  # Convert to km/h
					D[tracker_id].append(speed)
					#g = mode(Gender[tracker_id])
					#a = mode(Age[tracker_id])

    	    # Annotate frame
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			af = frame.copy()
			for tid in D :
				p1,p2,P,h1,h2,g,s = D[tid]
				X,Y = P
				gender,age = g
				Gl = list(Gender[tid])
				Al = list(Age[tid])
				gender = max(set(Gl), key=Gl.count)
				age = max(set(Al), key=Al.count)
				if gender is None:gender=""
				if age is None:age=""
				af = cv2.rectangle(af,p1,p2,(0,255,0),2)
				af = cv2.rectangle(af,h1,h2,(255,0,0),2)
				if s is not None:s=round(s,3)
				if s is None:v=""
				else:v=f"{s}km/h"
				#af = cv2.putText(af,f"#{tid}, v: {s} km/h , x: {100*X} cm, y: {100*Y} cm",p1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
				af = cv2.putText(af,f"#{tid}|{v}|{age}|{gender}",p1,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)
				#af = cv2.putText(af,f"{age}|{gender}",p1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
			sink.write_frame(af)

    	    #if cv2.waitKey(1) & 0xFF == ord("q"):
    	    #    break

			frames_done += 1
			print(f"{100 * frames_done/video_info.total_frames} % done")
			#if frames_done >= 0.5 * video_info.total_frames:break

	
	
	
def detectest():
	global head,head2
	#imgsrc = "sample.jpg"
	#I = cv2.imread(imgsrc)
	setVideo()
	for i in range(200):next(frame_generator)
	I = next(frame_generator)
	I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
	fulldetect(I,True,head)
	
if __name__ == "__main__":
	#headtest()
	#detectest()
	markVideo("videos/Corridor.mp4")
