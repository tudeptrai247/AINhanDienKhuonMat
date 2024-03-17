import cv2
import face_recognition

imgElon = face_recognition.load_image_file("pic/elon musk.jpg")
imgElon =cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgCheck = face_recognition.load_image_file("pic/elon check.jpg")
imgCheck =cv2.cvtColor(imgCheck,cv2.COLOR_BGR2RGB)

# xác định vị trí
faceLoc = face_recognition.face_locations(imgElon)[0]
print(faceLoc) #(Y1,X2,Y2,X1)

# Mã Hóa Hình Ảnh
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceCheck = face_recognition.face_locations(imgCheck)[0]
encodeCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck,(faceLoc[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(255,0,255),2)

results =face_recognition.compare_faces([encodeElon] , encodeCheck)
print(results)

## khi có nhiều bưc ảnh cần phải biết sai số là bao nhiêu
faceDiss =face_recognition.face_distance([encodeElon],encodeCheck)
print(results,faceDiss)
cv2.putText(imgCheck,f"{results}{1-(round(faceDiss[0],2))}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),2)


cv2.imshow("Elon",imgElon)
cv2.imshow("ElonCheck",imgCheck)
cv2.waitKey()