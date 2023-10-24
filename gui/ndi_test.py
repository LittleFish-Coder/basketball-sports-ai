import cvndi
import cv2

# 從 NDI 找到所有的 sources
sources = cvndi.get_sources()
# 回傳值為一個 list 大概如下：
# [DESKTOP-UB9TV1P (101), DESKTOP-UB9TV1P (103), DESKTOP-UB9TV1P (104), DESKTOP-UB9TV1P (105), DESKTOP-UB9TV1P (106), DESKTOP-UB9TV1P (107)]

# 列出所有的 sources
for source in sources:
    print(source.ndi_name)  # source是一個object，用ndi_name取得名稱，i.e. DESKTOP-UB9TV1P (105)

print(cvndi.ip_source(sources, '105'))  # 如果105在sources裡面，就會回傳source object，否則回傳None

# cvndi.VideoCapture()的參數是一個source object
# 我們把編號105的source傳進去
cap = cvndi.VideoCapture(cvndi.ip_source(sources, '105'))   

# 這邊就跟一般的cv2.VideoCapture()一樣了
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()