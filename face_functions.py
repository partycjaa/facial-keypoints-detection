import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import time

# załadowanie danych
def load_data(df):
    X = []
    for pixels in df.Image.values:
        pixels_arr = np.fromstring(pixels, sep=' ')
        image = pixels_arr.reshape(96,96)
        X.append(image/255)
    X = np.array(X)
    X = X.reshape(-1,96,96,1)
    columns = df.columns[:-1]
    y = df[columns].values /96
    return X.astype(np.float32), y.astype(np.float32)

# rysowanie przykładowych twarzy z zaznaczonymi punktami kluczowymi
def plotFaces(X, y=None, y_pred=None, s=10):
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    ind = np.random.default_rng().integers(0, X.shape[0], (4))

    for i, ax in zip(ind, axes.flatten()):

        ax.imshow(X[i], cmap='gray')

        if(y is not None):
            keypoints = y[i]
            x_coords = keypoints[::2]
            y_coords = keypoints[1::2]
            ax.scatter(x_coords*96,y_coords*96, s=s, color='red')

        if(y_pred is not None):
            keypoints = y_pred[i]
            x_coords = keypoints[::2]
            y_coords = keypoints[1::2]
            ax.scatter(x_coords*96,y_coords*96, s=s, color='blue')

    plt.show()


# obracanie danych wejściowych
def rotate_image_and_keypoints(image, keypoints):

    angle_degrees = np.random.uniform(-45,45)
    angle_rad = np.deg2rad(angle_degrees)

    rotated_image = rotate(image, angle_degrees, reshape=False, order=1, mode='nearest')

    x = keypoints[::2] * 96
    y = keypoints[1::2] * 96

    x_centered = x - 48
    y_centered = y - 48

    x_new = x_centered * np.cos(-angle_rad) - y_centered * np.sin(-angle_rad)
    y_new = x_centered * np.sin(-angle_rad) + y_centered * np.cos(-angle_rad)

    x_rotated = x_new + 48
    y_rotated = y_new + 48

    new_keypoints = np.empty_like(keypoints)
    new_keypoints[::2] = x_rotated / 96
    new_keypoints[1::2] = y_rotated / 96

    return rotated_image, new_keypoints

# funkcja sprawdzającą prędkość czytania danych
def reading_time(dataset):
    start_time = time.time()
    n_examples = 0

    for batch in dataset:
        x_batch, y_batch = batch[0], batch[1]
        n_examples += x_batch.shape[0]

    elapsed = time.time() - start_time
    print(f"Czas odczytu: {elapsed:.2f}s")
    print(f"Liczba odczytanych przykładów: {n_examples}")
    print(f"Średnia prędkość odczytu: {n_examples / elapsed:.2f} przykładów/s")

# ładowanie testowych obrazków
def load_test_images(df):
    X = []
    for pixels in df.Image.values:
        pixels_arr = np.fromstring(pixels, sep=' ')
        image = pixels_arr.reshape(96,96)
        X.append(image/255)
    X = np.array(X)
    X = X.reshape(-1,96,96,1)
    return X.astype(np.float32)

# porównywanie modeli
def compare_models(X_test, models, labels):

    ind = np.random.default_rng().integers(0, X_test.shape[0], (1))
    fig, axes = plt.subplots(2,2, figsize=(10,10))

    for i, ax in enumerate(axes.flatten()):

        image = X_test[ind]
        ax.imshow(image.reshape(96,96), cmap='gray')

        keypoints = models[i].predict(image, verbose=0)[0]
        x_coords = keypoints[::2]
        y_coords = keypoints[1::2]
        ax.scatter(x_coords*96,y_coords*96, s=10, color='red')

        ax.set_title(labels[i])

    plt.show()

