import os
import sys
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

folder = sys.argv[1]
car_id_counter = 1  #Counter za jedinstvene identifikatore vozila
car_tracker = {}  # Rečnik za praćenje automobila
predicted_count =0


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

train_dir = os.path.join(folder, 'pictures')

pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)

pos_features = []
neg_features = []
labels = []

nbins = 9
cell_size = (8, 8)
block_size = (3, 3)

img = pos_imgs[0]  # Choose one image for HOG initialization
hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf_svm = SVC(kernel='linear', probability=True)
clf_svm.fit(x_train, y_train)


def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]

def process_image(image, step_size, window_size=(580, 300), threshold=0.8, nms_threshold=0.2):
    detections = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for y in range(0, image.shape[0], 60):
        for x in range(0, image.shape[1], step_size):
            this_window = (y, x)
            window = gray_image[y:y + window_size[1], x:x + window_size[0]]

            if window.shape == (window_size[1], window_size[0]):
                window = cv2.resize(window, (pos_imgs[1].shape[1], pos_imgs[1].shape[0]))
                window = cv2.cvtColor(window, cv2.COLOR_GRAY2BGR)
                score = classify_window(window)

                if score > threshold:
                    y, x = this_window
                    w, h = window_size
                    detections.append((score, (x, y, x + w, y + h)))

    # Apply non-maximum suppression
    detections = non_max_suppression(detections, nms_threshold)

    for score, bbox in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return detections

def non_max_suppression(detections, threshold):
    if len(detections) == 0:
        return []

    # Sort detections based on their confidence scores
    detections = sorted(detections, key=lambda x: x[0], reverse=True)

    # Initialize a list to store the final selected detections
    selected_detections = [detections[0]]

    for current_detection in detections[1:]:
        add_current_detection = True

        for final_detection in selected_detections:
            overlap = calculate_overlap(current_detection[1], final_detection[1])

            if overlap > threshold:
                add_current_detection = False
                break

        if add_current_detection:
            selected_detections.append(current_detection)

    return selected_detections

def calculate_overlap(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    # Calculate the area of intersection
    x_intersection = max(0, min(x2a, x2b) - max(x1a, x1b))
    y_intersection = max(0, min(y2a, y2b) - max(y1a, y1b))
    intersection_area = x_intersection * y_intersection

    # Calculate the area of both bounding boxes
    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)

    # Calculate the overlap ratio
    overlap_ratio = intersection_area / float(min(area1, area2))

    return overlap_ratio








# Detekcija crvene linije
def detect_red_line(img):
    # Pretvaranje slike u HSV prostor boja
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definisanje opsega crvene boje u HSV prostoru
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Segmentacija crvene boje
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    # Hough transformacija na binarnoj slici
    lines = cv2.HoughLinesP(image=mask, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]),
                            minLineLength=200, maxLineGap=20)

    if lines is not None:
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if y2 > y1:
                detected_lines.append((x1, y1, x2, y2))
            else:
                detected_lines.append((x2, y2, x1, y1))

        #     # Draw the detected lines on the image
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # print("Detected lines [[x1 y1 x2 y2]]: \n", detected_lines)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB before showing
        # plt.show()
        return detected_lines
    else:
        # print("No detected lines.")
        return None

def get_line_params(line_coords):
    x1,x2, y1, y2 = line_coords[0] 

    if float(y2) - float(y1) != 0:
        k = (float(y2) - float(y1)) / (float(x2) - float(x1))  
        n = k * float(-x1) + float(y1)
        return x1, y1, x2, y2  
    else:
        return x1, y1, x2, y2 






def has_crossed_line(car_position, line_coords):
    # print("Car Position:", car_position)
    # print("Line Coords:", line_coords)

    if len(car_position) == 4 and len(line_coords) == 4:
        x_car, y_car, x2_car, y2_car = car_position
        x1_line, _, _, y2_line = line_coords
        
        if x_car==0:
         x_car=x2_car-500
            

        # Check if the right side of the car has crossed the line
        # if x_car < x1_line and  x2_car-x1_line>380 and x2_car>x1_line and y_car < y2_line :
        if x2_car > x1_line and x2_car-x1_line<150 and x_car<x1_line  and y_car < y2_line :

            # print(f"Automobil je prešao crvenu liniju. x_car: {x_car},x2_car:{x2_car} y_car: {y_car}, lineX: {x1_line}, lineY: {y2_line}")
            return True
        else:
            # print(f"Automobil nije prešao crvenu liniju. x_car: {x_car}, y_car: {y_car}, lineX: {x1_line}, lineY: {y2_line}")
            return False
    else:
        # print("Neispravan tuple za automobil.")
        return False







def process_video(video_path):
    sum_of_cars = 0

    cap = cv2.VideoCapture(video_path)

    while True:
        grabbed, frame = cap.read()

        if not grabbed:
            break

        line_coords = detect_red_line(frame)

        if line_coords is None:
            continue


        cars_detections = process_image(frame, step_size=160)

        for score, car_position in cars_detections:
            # print("Car Position:", car_position)

            if has_crossed_line(car_position, get_line_params(line_coords)):
                sum_of_cars += 1
                
        
              

        # small_frame = cv2.resize(frame, (800, 600))

        # # Display the frame with detected cars and the red line
        # cv2.imshow("Frame", small_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    return sum_of_cars

# suma = process_video("data2/videos/segment_4.mp4")
# print("Izračunata suma: ", suma)

# Učitavanje podataka iz CSV-a
csv_file_path = os.path.join(folder, 'counts.csv')

# Čitanje CSV datoteke
df = pd.read_csv(csv_file_path)
# Inicijalizacija prazne liste za pohranu rezultata MAE
mae_results = []

# Petlja koja prolazi kroz svaki video
for index, row in df.iterrows():
    video_name = row['Naziv_videa']
    video_extension = ".mp4"

    # Dodajte nastavak ako ga već nema
    if not video_name.endswith(video_extension):
        video_name += video_extension

    true_count = row['Broj_kolizija']
    video_path = os.path.join(folder, 'videos', video_name)

    predicted_count = process_video(video_path)


    # Računanje MAE za trenutni video
    mae = mean_absolute_error([true_count], [predicted_count])
    mae_results.append(mae)
    
    # Ispis rezultata za trenutni video
    print(f'Video: {video_name}, Predvidjeni broj: {predicted_count}, Stvarni broj: {true_count}, MAE: {mae}'.encode('utf-8'))

# Ispis ukupnog MAE za sve slike
total_mae = np.mean(mae_results)
print(f'Ukupni MAE za sve videe: {total_mae}')


    