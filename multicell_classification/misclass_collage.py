import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

st = time.time()


def create_misclass_collage(image_dir, model_name, cols=5, save_path=None, img_size=(3, 3)):
    """
    Create and save a grid collage of all images in a folder.

    Parameters:
    - image_dir: Folder containing misclassified images for a model.
    - model_name: Title for the figure.
    - cols: Number of columns in the collage.
    - save_path: Where to save the collage image (optional).
    - img_size: Size of each subplot in inches (width, height).
    """
    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if
                   f.lower().endswith('png')]
    num_images = len(image_files)
    rows = math.ceil(num_images / cols)

    # fig, axes = plt.subplots(rows, cols, figsize=(cols * img_size[0], rows * img_size[1]))
    # axes = axes.flatten()
    #
    # for ax, img_path in zip(axes, image_files):
    #     img = mpimg.imread(img_path)
    #     ax.imshow(img)
    #     ax.axis('off')
    #     ax.set_title(os.path.basename(img_path), fontsize=8)
    #
    # # Hide any empty subplots
    # for i in range(len(image_files), len(axes)):
    #     axes[i].axis('off')
    #
    # plt.suptitle(f"Misclassified Samples – {model_name}", fontsize=16)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300)
    #     print(f"Saved collage to: {save_path}")
    # plt.show()

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))  # Increase image size
    axes = axes.flatten()

    for ax, img_path in zip(axes, image_files):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')  # remove axis
        # No titles since GT/Pred is in image

    # Hide unused axes
    for i in range(len(image_files), len(axes)):
        axes[i].axis('off')

    plt.subplots_adjust(
        left=0.01, right=0.99, top=0.92, bottom=0.01,  # tighter margins around figure
        wspace=0.01, hspace=0.01  # minimal spacing between images
    )

    plt.suptitle(f"Misclassified Samples – {model_name}", fontsize=14, y=0.96)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved collage to: {save_path}")
    plt.show()


num_cols = 7
# create_misclass_collage("appendix_misclass/ResNet", "ResNet", cols=num_cols, save_path="resnet_collage.png")
# create_misclass_collage("appendix_misclass/DenseNet", "DenseNet", cols=num_cols, save_path="dense_collage.png")
# create_misclass_collage("appendix_misclass/EfficientNet", "EfficientNet", cols=num_cols, save_path="effnet_collage.png")
# create_misclass_collage("appendix_misclass/MobileNetV2", "MobileNetV2", cols=num_cols, save_path="mobnet_collage.png")
# create_misclass_collage("appendix_misclass/SimpleCNN", "SimpleCNN", cols=num_cols, save_path="simplecnn_collage.png")

print('Runtime', time.time() - st)
