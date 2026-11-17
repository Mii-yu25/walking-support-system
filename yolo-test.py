from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")

result = model.predict("walking-support-system\Lena_img.jpg", save=True)

result_img = result[0].plot()

cv2.imshow("Lena_img", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()