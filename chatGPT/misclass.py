import cv2
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

st = time.time()


def save_all_misclassifications(df, preds, gt, save_dir='misclassified_images', model_name='ChatGPT4o'):
    """
    Saves all misclassified images with GT and predicted labels overlaid.

    :param preds: List of predicted class labels (e.g., 'Cluster' or 'Non-Cluster')
    :param gt: List of ground truth class labels (same format)
    :param save_dir: Folder where misclassified images will be saved
    """
    img_dir = '../multicell_classification/classify_dataset/test'
    os.makedirs(save_dir, exist_ok=True)

    folder_names = {'Cluster': 'cluster_revised', 'Non-Cluster': 'non_cluster_revised'}
    image_list = []
    for i, filename in enumerate(df['Filename'].tolist()):
        if os.path.exists(img_dir + '/' + folder_names[ground_truth[i]] + '/' + filename) is False:
            print(img_dir + '/' + folder_names[ground_truth[i]] + '/' + filename)
        else:
            image_list.append(img_dir + '/' + folder_names[ground_truth[i]] + '/' + filename)

    relevant_indices = [i for i in range(len(preds)) if preds[i] != gt[i]]

    for count, i in enumerate(relevant_indices):
        image_path = image_list[i]
        orig = cv2.imread(image_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_rgb = cv2.resize(orig_rgb, (224, 224))

        gt_label = gt[i]
        pred_label = preds[i]

        # overlay = img.copy()
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # annotated_img = cv2.putText(orig_rgb.copy(), title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.7, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(overlay, f"GT: {gt_label}", (10, 25), font, 0.6, (0, 255, 0), 2)
        # cv2.putText(overlay, f"Pred: {pred_label}", (10, 50), font, 0.6, (0, 0, 255), 2)
        #
        # save_path = os.path.join(save_dir, f"img_{count:03}_GT_{gt_label}_Pred_{pred_label}.png")
        # cv2.imwrite(save_path, overlay)
        #
        # img_path = test_ds.get_image_list()[idx]
        # orig = cv2.imread(img_path)
        # orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # Annotate the image with GT and predicted

        title = f"GT: {gt_label}, Pred: {pred_label}"
        annotated_img = cv2.putText(orig_rgb.copy(), title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 0, 0), 2, cv2.LINE_AA)


        save_path = os.path.join(save_dir, f"img_{count:03}_GT_{gt_label}_Pred_{pred_label}.png")
        cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(relevant_indices)} misclassified images to: {save_dir}")


import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_misclass_collage(image_dir, model_name, cols=5, save_dir=None, img_size=(2.5, 2.5),
                            max_images_per_collage=40):
    """
    Create and save one or more grid collages of images in a folder.

    Parameters:
    - image_dir: Folder containing misclassified images for a model.
    - model_name: Title prefix for the figure(s).
    - cols: Number of columns in each collage.
    - save_dir: Folder to save the collages (optional).
    - img_size: Tuple (width, height) for each image in inches.
    - max_images_per_collage: Max images per figure to avoid overcrowding.
    """
    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith('.png')]
    total_images = len(image_files)

    if total_images == 0:
        print("No images found.")
        return

    num_collages = math.ceil(total_images / max_images_per_collage)

    for part in range(num_collages):
        start = part * max_images_per_collage
        end = min(start + max_images_per_collage, total_images)
        subset = image_files[start:end]
        rows = math.ceil(len(subset) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * img_size[0], rows * img_size[1]))
        axes = axes.flatten()

        for ax, img_path in zip(axes, subset):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')

        # Hide unused axes
        for i in range(len(subset), len(axes)):
            axes[i].axis('off')

        plt.subplots_adjust(
            left=0.01, right=0.99, top=0.92, bottom=0.01,
            wspace=0.01, hspace=0.01
        )

        plt.suptitle(f"Misclassified Samples – {model_name} (Part {part + 1})", fontsize=14, y=0.96)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_collage_part_{part + 1}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Saved collage to: {save_path}")

        plt.show()


img_dir = '../multicell_classification/classify_dataset/test'
test_df = pd.read_csv('test_exports/ver4ses1_test.csv')
test_gt_cluster = \
    os.listdir('../multicell_classification/classify_dataset/test/cluster_revised')
test_gt_non = os.listdir('../multicell_classification/classify_dataset/test/non_cluster_revised')

df = test_df.dropna(axis=0, how='all')

df = df.drop(columns=['Batch_Start'])
df = df[~df['Filename'].str.contains('Batch', na=False)]
df = df[~df['Filename'].str.contains('Filename', na=False)]
df['Filename'] = df['Filename'].apply(lambda x: x + '.png' if '.png' not in x else x)
df['Class'] = df['Class'].apply(lambda x: 'Cluster' if x == ' Cluster' else 'Non-Cluster')
df['gt'] = df['Filename'].apply(lambda x: 'Cluster' if x in test_gt_cluster else 'Non-Cluster')

print(df)

ground_truth = df['gt'].tolist()
predictions = df['Class'].tolist()

# folder_names = {'Cluster': 'cluster_revised', 'Non-Cluster': 'non_cluster_revised'}
#
# for i, filename in enumerate(df['Filename'].tolist()):
#     if os.path.exists(img_dir+'/'+folder_names[ground_truth[i]]+'/'+filename) is False:
#         print(img_dir+'/'+folder_names[ground_truth[i]]+'/'+filename)

save_all_misclassifications(df=df, preds=predictions, gt=ground_truth)
num_cols = 7
create_misclass_collage("misclassified_images", "ChatGPT 4o", cols=num_cols)

print('Runtime', time.time() - st)
