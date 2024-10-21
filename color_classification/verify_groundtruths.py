import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time

st = time.time()
df = pd.read_csv('groundtruths.csv', index_col=False)


def plot_all_channels(df_row, name):
    image2 = cv2.imread(df_row['chan2'])
    image3 = cv2.imread(df_row['chan3'])
    image7 = cv2.imread(df_row['chan7'])
    image11 = cv2.imread(df_row['chan11'])

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.title(f'Label: {name}')
    # Plot each image
    axs[0, 0].imshow(image2)
    axs[0, 1].imshow(image3)
    axs[1, 0].imshow(image7)
    axs[1, 1].imshow(image11)
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()

    plt.show()


for i in range(len(df)):
    # print(df.iloc[i]['chan2'])
    if i in range(500, 520):
        print(i)
        plot_all_channels(df.iloc[i], df.iloc[i]['label'])


print('Time taken:', time.time() - st)
