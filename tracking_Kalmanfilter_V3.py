## Kalman filter tracking of particles in a video V3 with filtering and adaptive thresholding
import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Parameters
video_file_name = 'video_phase_RBC_M_40.avi'
folder = r'D:\maria\Escritorio\Universidad\Maestria\Proyecto\Tracking\Simulation_Raul\videos'
movie_full_path = os.path.join(folder, video_file_name)
dxy = 3.75  # um
MO = 40     
pixel_size = dxy / MO
diam_um = 2.0  # um, diameter of the particles

# Initialize video capture and read the first frame
cap = cv2.VideoCapture(movie_full_path)
ret, old_frame = cap.read()
if not ret:
    raise ValueError("Could not read the first frame of the video.")

def detectar_centroides(frame_gray, pixel_size, diam_um):
    # speckle filtering
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    #blurred = cv2.bilateralFilter(frame_gray,9,75,75)

    # Calculate area in pixels
    #diam_px = diam_um / pixel_size_um
    #area_px = np.pi * (diam_px / 2) ** 2

    # configuration of SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    # Area filter
    params.filterByArea = True
    params.minArea = 600
    #params.maxArea = 600

    #params.minThreshold = 40
    #params.maxThreshold = 200

    # circularity
    params.filterByCircularity = False
    params.minCircularity = 1.0  # 1.0 is perfect circle

    # elongación 
    params.filterByInertia = True
    params.filterByConvexity = False

    # depending on the video, you may need to adjust these parameters
    params.filterByColor = False
    params.blobColor = 255  # 0 = oscuro, 255 = claro

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detected blobs
    keypoints = detector.detect(blurred)

    # Draw circles 
    debug_img = cv2.drawKeypoints(blurred, keypoints, None, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("Blobs detectados", debug_img)
    #cv2.waitKey(1)

    # Centroides
    centroids = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return centroids

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
centros = detectar_centroides(old_gray, pixel_size, diam_um)

if len(centros) == 0:
    raise ValueError("Could not read the first frame of the video.")

p0 = np.array(centros, dtype=np.float32)

# Create mask for drawing
mask = np.zeros_like(old_frame)

# ---- Configure Kalman filters for each point ----
class KalmanFilter2D:
    def __init__(self, x, y):
        # State: [x, y, vx, vy]
        self.state = np.array([x, y, 0, 0], dtype=np.float32)

        # Transition and observation matrices
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.P = np.eye(4, dtype=np.float32) * 100  # high initial uncertainty
        self.Q = np.eye(4, dtype=np.float32) * 0.01  # small process noise
        self.R = np.eye(2, dtype=np.float32) * 1    # measurement noise

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

# Create initial list of Kalman filters
kalman_filters = [KalmanFilter2D(x, y) for x, y in p0]

# Save trajectories with positions estimated by Kalman
trajectories = [ [kf.state[:2].copy()] for kf in kalman_filters ]
detected_positions = [[kf.state[:2].copy()] for kf in kalman_filters]
# Contador de cuadros sin detección
skipped_counts = [0] * len(kalman_filters)

# Main tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    centros_actuales = detectar_centroides(frame_gray, pixel_size, diam_um)

    p1 = np.array(centros_actuales, dtype=np.float32)

    # Predict with Kalman for each particle
    predictions = np.array([kf.predict() for kf in kalman_filters])

    # If no detections, update only prediction and save trajectory
    if len(p1) == 0:
        for i, kf in enumerate(kalman_filters):
            trajectories[i].append(kf.state[:2].copy())
            detected_positions[i].append(detected_positions[i][-1])
            skipped_counts[i] += 1
        continue

    # Distance matrix between predictions and detections
    dist_matrix = np.linalg.norm(predictions[:, None, :] - p1[None, :, :], axis=2)

    # Assign using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    max_dist = 55  # pixels, adjust according to expected velocity
    assigned_pred = set()
    assigned_det = set()

    # Update Kalman with assigned detections
    for r, c in zip(row_ind, col_ind):
        
        if dist_matrix[r, c] < max_dist:
            kalman_filters[r].update(p1[c])
            assigned_pred.add(r)
            assigned_det.add(c)
            detected_positions[r].append(p1[c].copy())
            skipped_counts[r] = 0
        else:
            # If distance not met, prediction without update
            detected_positions[r].append(detected_positions[r][-1])
            skipped_counts[r] += 1

    # For unassigned Kalman filters, only predict (without update)
    for i, kf in enumerate(kalman_filters):
        if i not in assigned_pred:
            detected_positions[i].append(detected_positions[i][-1])
            skipped_counts[i] += 1
            # Already predicted above, just add the prediction as estimation

    # Add new trackers for unassigned detections
    nuevas_detecciones = [p1[i] for i in range(len(p1)) if i not in assigned_det]
    for det in nuevas_detecciones:
        new_kf = KalmanFilter2D(det[0], det[1])
        kalman_filters.append(new_kf)
        trajectories.append([det.copy()])
        detected_positions.append([det.copy()])
        skipped_counts.append(0)

    # Remove trackers that skipped too long
    max_skips = 10
    for i in reversed(range(len(kalman_filters))):
        if skipped_counts[i] > max_skips:
            del kalman_filters[i]
            del trajectories[i]
            del detected_positions[i]
            del skipped_counts[i]

    # Save updated positions in trajectories
    for i, kf in enumerate(kalman_filters):
        trajectories[i].append(kf.state[:2].copy())

    # Draw
    #mask = np.zeros_like(old_frame)
    for traj in trajectories:
        if len(traj) > 1:
            pts = np.array(traj[-2:], dtype=int)
            cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
            cv2.circle(frame, tuple(pts[1]), 3, (0, 0, 255), -1)

    output = cv2.add(frame, mask)
    scale = 0.7  
    resized_output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Kalman Tracking", resized_output)
    #cv2.imshow("Kalman Tracking", output)

    # to exit the loop
    # Press 'Esc' key to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Plot trajectories (convert px to um)
plt.figure(figsize=(8, 6))
for i in range(len(detected_positions)):
    traj_det = np.array(detected_positions[i]) * pixel_size
    plt.plot(traj_det[:, 0], traj_det[:, 1], marker='o', markersize=3, alpha=0.5, label=f'Detection {i}')

plt.gca().invert_yaxis()
plt.xlabel('X (µm)')
plt.ylabel('Y (µm)')
plt.title('Real positions Detected')
plt.legend()
plt.grid(True)
plt.show()

#plt.figure(figsize=(12, 10))
#for i in range(len(kalman_filters)):
#    traj_kf = np.array(trajectories[i]) * pixel_size
#    traj_det = np.array(detected_positions[i]) * pixel_size
#    plt.plot(traj_kf[:, 0], traj_kf[:, 1], label=f'Kalman {i}')
#    plt.scatter(traj_det[:, 0], traj_det[:, 1], s=10, alpha=0.5)

#plt.gca().invert_yaxis()
#plt.xlabel('X (µm)')
#plt.ylabel('Y (µm)')
#plt.title('Trajectories: Kalman (line) vs Detection (points)')
#plt.legend()
#plt.grid(True)
#plt.show()
