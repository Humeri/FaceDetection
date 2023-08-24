import cv2
import mediapipe as mp

cap = cv2.VideoCapture("video3.mp4")

#medipipe kütüp. ait face_detection modülünü çağırıyoruz.
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.20) #fonksiyon çağırdık. fonk.içindeki değer takip parametresidir. 0 ve 1arasında en idealidir.
mpDraw = mp.solutions.drawing_utils



while True:
    success,img=cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = faceDetection.process(imgRGB)
    #print(results.detections) #bununla kordinat noktalarını görüyoruz.
    
    if results.detections: #eğer tespit varsa #enumarete 2 parametre alır.
        for id , detection in enumerate(results.detections): #detections içindekileri detectiona değişkenleri de id e
            bboxC = detection.location_data.relative_bounding_box
            #print(bboxC) #kutu için kordinatları gördük.
            
            #kutu oluşturma
            h,w,_ =img.shape
            bbox = int(bboxC.xmin*w) , int(bboxC.ymin*h) , int(bboxC.width*w) , int(bboxC.height*h)
            #print(bbox) #bbox kordinatları x,y,genişlik,yükseklik
            cv2.rectangle(img,bbox,(0,255,255),3) #rectangle oto resmi ve kordinatları alıyor. kutu rengini belirliyoruz. 3 değeri de kalınlık.
            
     



    cv2.imshow("img",img)
    cv2.waitKey(10)
    
    


