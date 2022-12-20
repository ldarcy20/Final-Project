import cv2

# Save Video
result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         120, (6000, 4000))
for i in range(374):
    temp = cv2.imread(f'res/{i}.JPG')
    print(temp)
    result.write(cv2.imread(f'res/{i}.JPG'))

result.release()
