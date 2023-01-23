import os
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import requests
import matplotlib.image as mpimg
from sklearn import decomposition
import matplotlib.pyplot as plt
import itertools
import seaborn as sn
import json
from shapely import wkt
import ast
from sklearn.preprocessing import OrdinalEncoder

def preprocess_features_dataset(df):
    columns_to_drop = ['edifc_stat', 'edifc_ty', 'points_geometry', 'footprint_geometry', 'points', 'edifc_uso']
    edifc_uso_to_include = ['commerciale', 'industriale', 'residenziale', 'servizio pubblico']

    with open("data/utils/edifc_uso_general.json", "r") as f:
        edifc_uso_mapping = json.load(f) 

    df["edifc_uso_desc"] = df["edifc_uso"].map(edifc_uso_mapping)
    return  df[df["edifc_uso_desc"].isin(edifc_uso_to_include)].drop(columns=columns_to_drop).dropna()


def preprocess_complete_dataset(df):
    columns_to_drop = ['edifc_stat', 'edifc_ty', 'points_geometry']
    edifc_uso_to_include = ['commerciale', 'industriale', 'residenziale', 'servizio pubblico']

    with open("data/utils/edifc_uso_general.json", "r") as f:
        edifc_uso_mapping = json.load(f) 

    df["edifc_uso_desc"] = df["edifc_uso"].map(edifc_uso_mapping)
    df['points_geometry'] = df.points_geometry.apply(wkt.loads)
    df['footprint_geometry'] = df.footprint_geometry.apply(wkt.loads)
    df['points'] = df.points.apply(lambda x: ast.literal_eval(x))

    return  df[df["edifc_uso_desc"].isin(edifc_uso_to_include)].drop(columns=columns_to_drop).dropna()


def load_and_preprocess_multiple_csv_from_path(path, preprocess_function=lambda x: x, only_first_n=None, **kwargs):
    files = [f for f in os.listdir(path) if f.endswith("csv")]
    files_to_load = files if only_first_n == None else files[:only_first_n]

    n = len(files_to_load)
    dfs = []
    for i, file in enumerate(files_to_load):
        print(f"Loading file {i}/{n}", end="\r", flush=True)
        try:
            filepath = os.path.join(path, file)
            df = pd.read_csv(filepath, **kwargs)
            filtered_df = preprocess_function(df)
            dfs.append(filtered_df)
        except:
            print(f"File named {file} not loaded properly!")
    return pd.concat(dfs)

def interpolate_points(x, y, z, num=200):
    x_linspace = np.linspace(min(x), max(x), num=num)
    y_linspace = np.linspace(min(y), max(y), num=num)

    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), z)

    return x_grid, y_grid, interp(x_grid, y_grid)

def plot_points(x, y, z, ax, title=None):
    ax.scatter(x, y, c=z, s=.15)
    ax.axis("equal")
    ax.plot(np.median(x), np.median(y), color='red', marker='o', markersize=20)
    if title != None:
        ax.set_title(title)


def plot_multipoly(multipoly, ax):
    for poly in multipoly.geoms:
            ax.plot(*poly.exterior.xy, color="red", linewidth=3)


def show_satellite_image(x, y, ax, zoom = 19, title=None):
    # Mapbox access token (TODO as a secret)
    access_token = 'pk.eyJ1IjoibWF0dGVvYmlnbGlvbGkiLCJhIjoiY2txcGN0cmJ5MDBqdTJvazV6cXdiM2ZqOSJ9.Bd6Gd05464fMSOpqCB-uTw'

    # Define the Mapbox API endpoint for creating a static map
    center_x, center_y = np.median(x), np.median(y)
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/pin-l+f41010({center_x},{center_y})/{center_x},{center_y},{zoom},0/512x512@2x?access_token={access_token}"

    # Send the API request and get the image
    with open("satellite_image.jpeg", 'wb') as f:
            f.write(requests.get(url).content)

    satellite_image = mpimg.imread('satellite_image.jpeg', format="jpeg")
    ax.imshow(satellite_image)
    if title != None:
        ax.set_title(title)

def show_dataset_sample(points, footprint=None, figsize=(20, 8)):
    x, y, z = map(np.array, zip(*points))
    interp_x, interp_y, interp_z = interpolate_points(x, y, z)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    plot_points(x, y, z, ax[0], title="Raw Points")
    plot_points(interp_x, interp_y, interp_z, ax[1], title="Interpolated Points")
    show_satellite_image(x, y, ax[2], title="Satellite Image")

    if footprint is not None:
        plot_multipoly(footprint, ax[0])
        plot_multipoly(footprint, ax[1])

    plt.draw()

def balance_dataset(dataset, column_name):
    column_sizes = dataset.groupby([column_name]).size()
    min_n_rows = min(column_sizes)
    col_values = list(column_sizes.index)

    dfs = []
    for edifc_uso in col_values:
        dfs.append(dataset[dataset[column_name] == edifc_uso].sample(n=min_n_rows))
    return pd.concat(dfs)

def plot_vectors(X, y, encoder, title="Plot", figsize=(20, 10)):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X) # Compute PCA
    X_pca = pca.transform(X) # Project data onto first two principal components

    # Visualize data
    plt.subplots(figsize=figsize)
    for label, encoding in encoder.items():
        indexes = np.array(y) == encoding
        plt.scatter(X_pca[indexes][:,0], X_pca[indexes][:,1], s=1, label=label)
    plt.title(title, {'fontsize': 15})
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, labels_encoder,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = sorted(labels_encoder, key=labels_encoder.get)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show_dataset_balance(dataset, column_name):
    sn.countplot(x = column_name, data = dataset, palette="Set2") 
    plt.xticks(rotation=45)
    plt.show()

def show_data_example(dataset, example_n): 
    points = list(dataset["points"])[example_n]
    if "footprint_geometry" in dataset.columns:
        footprint = list(dataset["footprint_geometry"])[example_n]
        show_dataset_sample(points, footprint)
    else:
        show_dataset_sample(points)

def multiclass_label_encoder(labels):

    # Define mapping and encode labels
    encoder = {label: encoding for encoding, label in enumerate(np.unique(labels))}
    y = [encoder[label] for label in labels]

    return y, encoder