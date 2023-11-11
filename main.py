import cv2
import numpy as np
import lanes

video = cv2.VideoCapture("./video/solidWhiteRight.mp4")

fps = int(video.get(cv2.CAP_PROP_FPS))

while video.isOpened():
    ret, frame = video.read()
    
    if ret == True:
        canny_edges = lanes.canny_edge_detector(frame)
        cropped_image = lanes.ROI_mask(canny_edges)
        
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=2,              # Distance resolution in pixels
            theta=np.pi / 180,  # Angle resolution in radians
            threshold=100,      # Min. number of intersecting points to detect a line  
            lines=np.array([]), # Vector to return start and end points of the lines indicated by [x1, y1, x2, y2] 
            minLineLength=40,   # Line segments shorter than this are rejected
            maxLineGap=25       # Max gap allowed between points on the same line
        )
        
        # Визуализация
        averaged_lines = lanes.avg_lines (frame, lines)              # Усредните линии Хафа.
        combined_image = lanes.draw_lines(frame, averaged_lines, 5)  # Добавление усредненных линий на исходное изображение

        cv2.imshow("Video", combined_image)
        
        

        if cv2.waitKey(fps) & 0xFF == ord('q'):
            break
    else: 
        break

video.release()
cv2.destroyAllWindows()