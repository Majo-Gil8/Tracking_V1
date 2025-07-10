import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, x, y, P_init, Q, R):
        self.state = np.array([x, y, 0, 0], dtype=np.float32)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * P_init
        self.Q = np.eye(4, dtype=np.float32) * Q
        self.R = np.eye(2, dtype=np.float32) * R

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

def detectar_centroides(frame_gray, minArea, maxArea, blobColor, filter_type,
                        filterByCircularity, filterByInertia, filterByConvexity, filterByColor):
    # Apply the selected filter
    if filter_type == "gaussian":
        filtered = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    elif filter_type == "bilateral":
        filtered = cv2.bilateralFilter(frame_gray, 9, 75, 75)
    else:
        raise ValueError("filter_type must be 'gaussian' or 'bilateral'")

    # Configure blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = minArea
    params.maxArea = maxArea

    params.filterByCircularity = filterByCircularity

    params.filterByInertia = filterByInertia
    params.filterByConvexity = filterByConvexity

    params.filterByColor = filterByColor
    params.blobColor = blobColor

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(filtered)
    centroids = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return centroids

def kalman_tracking_video(
    video_path,
    pixel_size,
    minArea,
    maxArea,
    blobColor,
    filter_type,
    filterByCircularity,
    filterByInertia,
    filterByConvexity,
    filterByColor,
    max_dist,
    max_skips,
    scale,
    P_init,
    Q_val,
    R_val,
    show_window,
    show_plot
):

    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame of the video.")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = detectar_centroides(old_gray, minArea, maxArea, blobColor, filter_type,
                              filterByCircularity, filterByInertia, filterByConvexity,
                              filterByColor)
    if len(p0) == 0:
        raise ValueError("No se detectaron partículas en el primer frame.")

    kalman_filters = [KalmanFilter2D(x, y, P_init, Q_val, R_val) for x, y in p0]
    trajectories = [[kf.state[:2].copy()] for kf in kalman_filters]
    detected_positions = [[kf.state[:2].copy()] for kf in kalman_filters]
    skipped_counts = [0] * len(kalman_filters)
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1 = detectar_centroides(frame_gray, minArea, maxArea, blobColor, filter_type,
                                 filterByCircularity, filterByInertia, filterByConvexity,
                                 filterByColor)
        predictions = np.array([kf.predict() for kf in kalman_filters])

        if len(p1) == 0:
            for i, kf in enumerate(kalman_filters):
                trajectories[i].append(kf.state[:2].copy())
                detected_positions[i].append(detected_positions[i][-1])
                skipped_counts[i] += 1
            continue

        dist_matrix = np.linalg.norm(predictions[:, None, :] - p1[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        assigned_pred = set()
        assigned_det = set()

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < max_dist:
                kalman_filters[r].update(p1[c])
                assigned_pred.add(r)
                assigned_det.add(c)
                detected_positions[r].append(p1[c].copy())
                skipped_counts[r] = 0
            else:
                detected_positions[r].append(detected_positions[r][-1])
                skipped_counts[r] += 1

        for i in range(len(kalman_filters)):
            if i not in assigned_pred:
                detected_positions[i].append(detected_positions[i][-1])
                skipped_counts[i] += 1

        nuevas_detecciones = [p1[i] for i in range(len(p1)) if i not in assigned_det]
        for det in nuevas_detecciones:
            new_kf = KalmanFilter2D(det[0], det[1], P_init, Q_val, R_val)
            kalman_filters.append(new_kf)
            trajectories.append([det.copy()])
            detected_positions.append([det.copy()])
            skipped_counts.append(0)

        for i in reversed(range(len(kalman_filters))):
            if skipped_counts[i] > max_skips:
                del kalman_filters[i]
                del trajectories[i]
                del detected_positions[i]
                del skipped_counts[i]

        for i, kf in enumerate(kalman_filters):
            trajectories[i].append(kf.state[:2].copy())

        for traj in trajectories:
            if len(traj) > 1:
                pts = np.array(traj[-2:], dtype=int)
                cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                cv2.circle(frame, tuple(pts[1]), 3, (0, 0, 255), -1)

        output = cv2.add(frame, mask)
        resized_output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if show_window:
            cv2.imshow("Kalman Tracking", resized_output)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    if show_plot:
        # --- PRIMER PLOT: Trayectorias ---
        fig, ax = plt.subplots(figsize=(8, 6))
        labels_text = []
        num_detections = len(detected_positions)
        cmap = plt.cm.get_cmap('tab20', num_detections)  # Colores consistentes

        for i in range(num_detections):
            traj_det = np.array(detected_positions[i]) * pixel_size
            ax.plot(traj_det[:, 0], traj_det[:, 1], marker='o', markersize=3, alpha=0.7, color=cmap(i))
            labels_text.append(f'Detection {i}')

        ax.invert_yaxis()
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title('Trayectory')
        ax.grid(True)
        plt.tight_layout()

        # Mostrar sin bloquear
        plt.show(block=False)

        # --- SEGUNDO PLOT: Leyenda en gráfico aparte ---
        fig_legend, ax_legend = plt.subplots(figsize=(4, num_detections * 0.3))
        ax_legend.axis('off')

        for i in range(num_detections):
            ax_legend.plot([], [], marker='o', color=cmap(i), label=f'Detection {i}')

        ax_legend.legend(loc='center left', frameon=True)
        plt.title("Trackers Legend")
        plt.tight_layout()
        plt.show()


    return detected_positions, trajectories

if __name__ == "__main__":
    folder = r'D:\maria\Escritorio\Universidad\Maestria\Proyecto\Tracking\Simulation_Raul\videos'
    video_file_name = 'video_phase_RBC_10x.mp4'
    movie_full_path = os.path.join(folder, video_file_name)

    dxy = 3.75  # µm
    MO = 40
    pixel_size = dxy / MO

    kalman_tracking_video(
        video_path=movie_full_path,
        pixel_size=pixel_size,
        minArea=150,
        maxArea=500,
        blobColor=255,
        filter_type='gaussian',  # 'gaussian' o 'bilateral'
        filterByCircularity=False,
        filterByInertia=False,
        filterByConvexity=False,
        filterByColor=True,
        max_dist=40,
        max_skips=0,
        scale=0.7, # for resizing the output window of the video its' not necesary in the interface i think so
        P_init=100,
        Q_val=0.1,
        R_val=1.0,
        show_window=True,
        show_plot=True
    )
