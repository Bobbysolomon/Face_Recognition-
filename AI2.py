import numpy as n
import cv2
import face_recognition as f
import os


pa=r'E:/B/Bob'
im=[]
cn=[]
f1=""
d=os.listdir(pa)
for i in d:
    d=cv2.imread(f'{pa}/{i}')
    im.append(d)
    cn.append(i.split(".")[0])
def encod(im):
    e=[]
    for i in im:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        e1=f.face_encodings(i)[0]
        e.append(e1)
    return e

f2=encod(im)
print("Encoding Complete")

d=cv2.VideoCapture(0)

while(True):
    s,ib=d.read()
    q=cv2.resize(ib,(0,0),None,0.25,0.25)
    q=cv2.cvtColor(q,cv2.COLOR_BGR2RGB)

    w=f.face_locations(q)
    eq=f.face_encodings(q,w)

    for z,x in zip(eq,w):
        mat=f.compare_faces(f2,z)
        fdis=f.face_distance(f2,z)
        mn=n.argmin(fdis)

        if mat[mn]:
            nam=cn[mn].upper()
            print(nam)
            y1,x2,y2,x1 = x
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(ib,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(ib,(x1,y1-35),(x2,y2),(0,255,0))
            cv2.putText(ib,nam,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow("web",ib)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break











