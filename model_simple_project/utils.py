import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms
from PIL import Image
from google.cloud import storage

bucket = storage.Client().bucket("cosi149p1")
coord_txt_blob = bucket.blob("coordinates_train.csv")
coord_txt_bytes = coord_txt_blob.download_as_string()
coord_txt = pd.read_csv(io.BytesIO(coord_txt_bytes))

tfs = transforms.Compose([transforms.ToTensor()])

# Grab image from GCS
# Usage: grab_image("train/P_NCALM_5_00001.tif")
# Returns: Tensor
def grab_image(filepath):
    filepath = filepath.strip()
    blob = bucket.blob(filepath)
    obj_bytes = blob.download_as_string()
    im = Image.open(io.BytesIO(obj_bytes))
    return tfs(im)[0]

# Plot tensor in 3d
def plot_3d(ts, filepath, class_num):
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    ha.set_title(f"{filepath}, class {class_num}")
    X, Y = np.meshgrid(range(129), range(129))
    ha.plot_surface(X, Y, ts.numpy())
    plt.show()

# Plot tensor in heatmap
def heatmap(ts, filepath, class_num):
    ax = sns.heatmap(ts.numpy())
    ax.invert_yaxis()
    plt.title(f"{filepath}, class {class_num}")
    plt.show()

# Grab show n random negative images in class 0
def sample_neg_visualize(n):
    file_names = coord_txt.loc[coord_txt[' class'] == 0].sample(n)[' filename']
    for _, f in file_names.items():
        ts = grab_image(f)
        plot_3d(ts, f, 0)
        heatmap(ts, f, 0)

# Grab show n random positive images in class 2
def sample_pos2_visualize(n):
    file_names = coord_txt.loc[coord_txt[' class'] == 2].sample(n)[' filename']
    for _, f in file_names.items():
        ts = grab_image(f)
        plot_3d(ts, f, 2)
        heatmap(ts, f, 2)

# Grab show n random positive images in class 3
def sample_pos3_visualize(n):
    file_names = coord_txt.loc[coord_txt[' class'] == 3].sample(n)[' filename']
    for _, f in file_names.items():
        ts = grab_image(f)
        plot_3d(ts, f, 3)
        heatmap(ts, f, 3)
