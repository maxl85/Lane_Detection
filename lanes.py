import cv2
import numpy as np


# Возвращает края, найденые на изображении
def canny_edge_detector(frame):
    
    # Преобразуем исходное изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Для уменьшения шума применяем размытие по Гаусу 5x5
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Применяем детектор краев Canny с параметрами minVal=50 и maxVal=150
    canny = cv2.Canny(blur, 50, 150)
    
    return canny

# Накладывает маску на изображение, оставляя только полосу дороги
def ROI_mask(image):
    
    # Берем высоту и ширину изображения
    height = image.shape[0]
    width = image.shape[1]

    
    # Определяем координаты треугольника для выделения области полосы движения и удаления других ненужных частей изображения.
    polygons = np.array([ 
        [(100, height), (round(width/2), round(height/2)), (920, height)] 
        ]) 
    
    # Создаем маску
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, polygons, 255)  ## 255 это цвет маски
    
    # Применяем побитовое логическое AND между изображением с краями (canny) и изображением маски
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def get_coordinates (image, params):
    slope, intercept = params 
    y1 = image.shape[0]     
    y2 = int(y1 * (3/5)) # Setting y2 at 3/5th from y1
    x1 = int((y1 - intercept) / slope) # Deriving from y = mx + c
    x2 = int((y2 - intercept) / slope) 
    
    return np.array([x1, y1, x2, y2])

# Returns averaged lines on left and right sides of the image
def avg_lines(image, lines): 
    
    left = [] 
    right = [] 
    
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4)
        # print(x1, y1, x2, y2)
          
        # Fit polynomial, find intercept and slope 
        params = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = params[0] 
        y_intercept = params[1] 
        
        # print(slope)
        if slope < 0: 
            left.append((slope, y_intercept)) #Negative slope = left lane
        else: 
            right.append((slope, y_intercept)) #Positive slope = right lane
    
    # Avg over all values for a single slope and y-intercept value for each line
    if left != []:
        left_avg = np.average(left, axis = 0) 
        left_line = get_coordinates(image, left_avg) 
    else:
        left_line = [0, 0, 0, 0]
    
    if right != []:
        right_avg = np.average(right, axis = 0) 
        right_line = get_coordinates(image, right_avg) 
    else:
        right_line = [0, 0, 0, 0]
  
    return np.array([left_line, right_line])


# Draws lines of given thickness over an image
def draw_lines(image, lines, thickness): 
    line_image = np.zeros_like(image)
    color=[0, 0, 255]
    
    if lines is not None: 
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

            
    # Merge the image with drawn lines onto the original.
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    
    return combined_image