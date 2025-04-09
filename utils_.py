import torch
from skimage import io
import os
import numpy as np
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from pathlib import Path



ROOT_DIR = Path(os.getcwd())
FONT_DIR = ROOT_DIR / r"movistar-text/Movistar Text Regular.ttf"


#### function to view bbox on image
def view_bboxes(image_id, dataframe, event_type=None, fold_dir=None, tiling=None, no_labels=False):
    
    if tiling:
         sample = io.imread(os.path.join(fold_dir, image_id + '.jpg'))
    
    else:
        sample = io.imread(os.path.join(fold_dir, image_id))

    # convert image to tensor
    if type(sample) != torch.Tensor:
        sample = torch.from_numpy(sample)
        #permute the image to C, H, W
        sample = sample.permute(2, 0, 1)

    image_data = dataframe[dataframe['Image_ID']==image_id].reset_index(drop=True)

    # create tensor of bbox in  (xmin, ymin, xmax, ymax) format
    bboxes = []
    text_labels = []
    for row in range(image_data.shape[0]):
        bbox = []
        xmin, ymin, xmax, ymax = image_data.loc[row, 'xmin'], image_data.loc[row, 'ymin'], image_data.loc[row, 'xmax'], image_data.loc[row, 'ymax']
        bboxes.append([xmin, ymin, xmax, ymax])

        if event_type=='eval':
            # save label too
            text_labels.append(f"{image_data.loc[row, 'class']} : {image_data.loc[row, 'confidence']:.2f}")
        else:
            text_labels.append((image_data.loc[row, 'class']))



    bboxes =torch.from_numpy(np.array(bboxes))


    if sample.shape[1] >1250:
        if no_labels:
            output_image = draw_bounding_boxes(sample, bboxes, colors="red", width=6, font=FONT_DIR, font_size=20)
        else:    
            output_image = draw_bounding_boxes(sample, bboxes, text_labels, colors="red", width=6, font=FONT_DIR, font_size=20)

    else:
        if no_labels:
            output_image = draw_bounding_boxes(sample, bboxes, colors="red", width=6, font=FONT_DIR, font_size=20)
        else:
            output_image = draw_bounding_boxes(sample, bboxes, text_labels, colors="red", width=6, font=FONT_DIR, font_size=5)

    
    output_image= output_image.permute(1, 2, 0)
    # plt.imshow(output_image, ax=ax)
    return output_image
    


def plot_images_against(selected_images:list, dataframe_original, dataframe_1, dataframe_2, images_fold_dir, dataframe_1_title, dataframe_2_title):

    for image_file in (selected_images):

        if dataframe_2 is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(25,12))
            ax3.set_title(f'{dataframe_2_title}: {image_file}', fontsize=20)
            ax3.imshow(view_bboxes(image_file, dataframe_2, event_type='eval', fold_dir=images_fold_dir))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(25,12))

        ax1.set_title(f'Ground Truth: {image_file}', fontsize=20)
        ax1.imshow(view_bboxes(image_file, dataframe_original, fold_dir=images_fold_dir))

        ax2.set_title(f'{dataframe_1_title}: {image_file}', fontsize=20)
        ax2.imshow(view_bboxes(image_file, dataframe_1, event_type='eval', fold_dir=images_fold_dir))

    
        
        

