import cv2
import face_recognition
import os

import numpy as np

#load ảnh từ kho nhận dạng
path ="pic"
images =[]
className =[]
myList = os.listdir(path)
print(myList) #load anh trong file pic
for cl in myList:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}") #chạy từng bước ảnh và ra từng đường dẫn
    images.append(curImg) #luu buc anh vao ma tran
    className.append(os.path.splitext(cl)[0]) #tách ten file ra , chon so 0 để lấy tên
print(len(images)) #in ra số lượng
print(className) # in ra tên

#mã hóa và xác định vị trí
def Mahoa(images):
    encodeList =[] #ds rỗng để chua các mã hóa ảnh khuôn mặt
    for img in images: # chạy ra từng ma trận 1 cua bức ảnh
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ##chuyển bgr sang rgb , vi opencv chỉ nhận rgb
        encode =face_recognition.face_encodings(img)[0] # mã hóa chạy tung buc anh mà chi lay 1 buc anh nen là img[0]
        encodeList.append(encode) #day vao encodelist luu lai gia tri ma hoa
    return encodeList

endcodeListKnow =  Mahoa(images)
print("Ma Hoa Thanh Cong")
print(len(endcodeListKnow))

#khoi dong cam
cap = cv2.VideoCapture("mr beast.mp4")  # 0 là chỉ số của camera mặc định trên máy tính

while True:
    ret, frame=cap.read() #frame là khung hình , ret là chuyển về giá trị true hay false
    frameS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5) #resize lai kich thuoc
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB) #chuyen lai sang thanh rgb

    #Xác định vị trí khuôn mặt và encode trên cam
    facecurFram= face_recognition.face_locations(frameS) # lấy từng khuôn mặt và vi trí hiện tại
    endcodecurFram =face_recognition.face_encodings(frameS) #encode mặt tại khuôn hình hiện tại

    for endcodeFace , faceLoc in zip(endcodecurFram , facecurFram): #lấy vị trí khuôn mặt và mã hóa theo cặp , endcodecurFram chứa mã hóa còn facecurFram chứa vị trí
        matches =face_recognition.compare_faces(endcodeListKnow,endcodeFace) #so sánh ảnh mã hóa trong kho với ảnh hiện tại
        faceDis = face_recognition.face_distance(endcodeListKnow,endcodeFace) #so sánh để xem khoảng cách bao xa
        print(faceDis)
        matchIndex = np.argmin(faceDis) #Lấy về giá tri nhỏ nhất , sd argmin để tìm

        if(faceDis[matchIndex]) <0.50:
            name = className[matchIndex].upper() #để lấy tên
        else:
            name = "Unknow"

        #in tên lên frame
        y1 , x2 ,y2 ,x1 = faceLoc
        y1, x2, y2, x1 = y1*2 , x2*2 ,y2*2 ,x1*2
        cv2.rectangle(frame,(x1 , y1) ,(x2 , y2),(0,255,0) ,2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #de tên lên frame

    cv2.imshow('AI nhan dien khuon mat', frame)
    if cv2.waitKey(1) == ord("q"): #bấm Q để thoat
        break
cap.release() # giải phóng camera
cv2.destroyAllWindows() # thoát tát cả cua sổ
