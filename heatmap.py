import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

# --------------------------------------------------
# This file contains code to create a heatmap analysis of 
# an AI trained model. The heatmap analysis will highlight
# specific areas that the AI looks at to make it's decision.
# --------------------------------------------------

def preprocess(dat):
    """
    Preprocesses the dat to match the parameters used in training.
    In this example, it is processed to optimize training on CT scans.
    
    Parameters:
    dat - numpy array of the data
    
    Returns:
    dat - preprocessed dat
    """
    
    dat = np.clip(dat, -150, 250) / 50
    
    return dat

def predict(dat, model):
    """
    Runs model prediction and returns the softmax prediction. This predict
    is used for a standard slice-by-slice model classifier. 
    Ex. Each slice within a 96x96x96 is classified as a class (96 outputs total)
    """
    # --- preprocess and predict dat
    dat = preprocess(dat)
    logits = model.predict(dat)
    
    # --- calculate softmax
    softmax = lambda x : np.exp(x) / np.exp(x).sum()
    output = softmax(model.predict(dat)).ravel()[-1]

    return output

def predict_2dunet(dat, model):
    
    # --- preprocess dat
    dat = preprocess(dat)
    logits = model.predict(dat)
    
    # --- calculate softmax and keep severe scores
    softmax = lambda x : np.exp(x) / np.exp(x).sum()
    output = softmax(model.predict(dat))[0, ..., 2:]
    
    return output

def create_map(dat, occ_shape=(1, 8, 8), occ_steps=(1, 4, 4), occ_value='min'):

    # --- get smallest value in data
    VALUE = getattr(np, 'min')(dat)

    # --- Create occlusion map
    shape = np.array(dat.shape[1:4]) / np.array(occ_steps)
    shape = np.floor(shape).clip(min=1).astype('int')
    preds = np.zeros(shape, dtype='float32')

    # --- Run inferences
    zz, yy, xx = shape
    zs, ys, xs = occ_steps
    zp, yp, xp = occ_shape

    for z in range(zz):
        for y in range(yy):
            for x in range(xx):

                printp('Creating occlusion map', np.count_nonzero(preds) / preds.size)

                # --- Create occlusion
                z0 = z * zs - int(zp / 2)
                y0 = y * ys - int(yp / 2)
                x0 = x * xs - int(xp / 2)

                z1 = z0 + zp 
                y1 = y0 + yp
                x1 = x0 + xp

                z0 = max(z0, 0)
                y0 = max(y0, 0)
                x0 = max(x0, 0)

                z1 = min(z1, dat_.shape[1])
                y1 = min(y1, dat_.shape[2])
                x1 = min(x1, dat_.shape[3])

                # --- block out values
                occ = dat_.copy()
                occ[:, z0:z1, y0:y1, x0:x1] = VALUE

                # --- Run inference
                preds[z, y, x] = predict(occ, model)

    # --- Resize back to original resolution
    zoom = np.array(dat.shape[1:4]) / shape
    preds = ndimage.zoom(preds, zoom, order=1)
    preds = np.expand_dims(preds, axis=-1)
    
    return preds

def create_png(dat, preds, outpath, vmin=-200, vmax=200):
    """
    Creates an overlay of the heatmap onto the original image and
    saves is as a png file.
    """
    
    # --- uses the matplot lib to visualize the figure
    fig = plt.figure(frameon=False, figsize=(6,6))
    plt.imshow(np.squeeze(dat), cmap=plt.cm.gray, vmin=vmin,vmax=vmax, interpolation='nearest')
    plt.imshow(np.squeeze(preds), cmap=plt.cm.viridis, alpha=0.3, interpolation='bilinear')
    plt.axis('off')
    
    # --- save figure into png
    plt.savefig(outpath + '.png')