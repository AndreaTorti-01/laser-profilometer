import cv2
import os

cap = cv2.VideoCapture(0) # change the number to switch between cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
wait_ms = int(500/fps)
num = 0

if not os.path.exists('images'):
    os.makedirs('images')
    print("images Directory created!")

print("cap resolution: " + str(int(cap.get(3))) + 'x' + str(int(cap.get(4))) + 'px' + ' at ' + str(fps) + 'fps')

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(wait_ms)

    if k == 27: # wait for 'esc' key to exit
        break
    elif k == ord('s'): # wait for 's' key to save
        cv2.imwrite('images/img' + str(num) + '.png', img)

        print("image saved!")
        num += 1

    cv2.imshow('Img',img)
    cv2.setWindowTitle('Img', 's to save image, esc to exit')

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()