import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_heatmaps(shape, landmarks, scale_w, scale_h):
    def gaussian_2d(shape, center, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32)
        ys = np.expand_dims(np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32), -1)
        alpha = -0.5 / (sigma**2)
        heatmap = np.exp(alpha * ((xs - center[0])**2 + (ys - center[1])**2))
        return heatmap

    heatmaps = []
    for i in range(0, len(landmarks), 2):
        x, y = float(landmarks[i])*scale_w, float(landmarks[i+1])*scale_h
        heatmaps.append(gaussian_2d(shape, (int(x), int(y)), sigma=5.0))
    return heatmaps

image_num = 4060
out_w, out_h = (192, 144)
image_path = '../TEyeD/train/image/000%d.png' % (image_num)
image = cv2.imread(image_path)
in_h, in_w, _ = image.shape
scale_w, scale_h = out_w/in_w, out_h/in_h 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (out_w, out_h))

iris_landmark = '../TEyeD/train/landmark/iris_landmark.txt'
lid_landmark = '../TEyeD/train/landmark/lid_landmark.txt'
pupil_landmark = '../TEyeD/train/landmark/pupil_landmark.txt'

iris_info = np.loadtxt(iris_landmark, delimiter=',', dtype=np.str)[image_num]
lid_info = np.loadtxt(lid_landmark, delimiter=',', dtype=np.str)[image_num]
pupil_info = np.loadtxt(pupil_landmark, delimiter=',', dtype=np.str)[image_num]

landmarks = np.concatenate([iris_info, lid_info, pupil_info])

plt.subplot(211)
heatmaps = get_heatmaps((out_w, out_h), landmarks, scale_w, scale_h)
heatmaps = np.array(heatmaps)
merged_heatmaps = np.mean(heatmaps, axis=0)
plt.imshow(merged_heatmaps)

plt.subplot(212)
for i in range(0, len(landmarks), 2):
    x, y = float(landmarks[i]) * scale_w, float(landmarks[i+1])* scale_h
    image = cv2.circle(image, (int(x),int(y)), 1, (0,0,255), 3)

plt.imshow(image)
plt.show()

