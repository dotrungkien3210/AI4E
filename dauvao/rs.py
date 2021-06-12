import cv2

image = cv2.imread('image.png')
print(image.shape)
cv2.imshow('anh',image)
cv2.waitKey()
crop_img = image[100, 100]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)