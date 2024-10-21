import random
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics


def result_visualizer(dataframe, labels, model_name, multi_channels=False, num_samples=5):
    max_length = len(dataframe)
    res = random.sample(range(max_length), num_samples)
    for i in res:
        file = dataframe['chan2'][i]
        file = file.replace('chan2', 'chan1')
        if multi_channels:
            c1 = cv2.imread(file)
            file = file.replace('chan1', 'chan2')
            c2 = cv2.imread(file)
            file = file.replace('chan2', 'chan3')
            c3 = cv2.imread(file)
            file = file.replace('chan3', 'chan7')
            c7 = cv2.imread(file)
            file = file.replace('chan7', 'chan11')
            c11 = cv2.imread(file)

            rows = 2
            cols = 3
            fig = plt.figure()
            fig.add_subplot(rows, cols, 1)
            plt.imshow(c1)
            plt.title('Channel 1')
            plt.axis('off')
            # plt.show()
            fig.add_subplot(rows, cols, 2)
            plt.imshow(c2)
            plt.title('Channel 2')
            plt.axis('off')
            # plt.show()
            fig.add_subplot(rows, cols, 3)
            plt.imshow(c3)
            plt.title('Channel 3')
            plt.axis('off')
            # plt.show()
            fig.add_subplot(rows, cols, 4)
            plt.imshow(c7)
            plt.title('Channel 7')
            plt.axis('off')
            # plt.show()
            fig.add_subplot(rows, cols, 5)
            plt.imshow(c11)
            plt.title('Channel 11')
            plt.axis('off')
            # plt.show()
            plt.suptitle(t=model_name + " Cluster: " + str(labels[i]))
            # plt.show()
        else:
            plt.imshow(cv2.imread(file))
            plt.title(label=model_name + " Cluster: " + str(labels[i]))
            # plt.show()
    plt.show()


def cluster_scoring(gt_labels, cluster_labels, model_name):
    print(f'-------------{model_name} Scores ------------------------')
    print(f"Homogeneity: {metrics.homogeneity_score(gt_labels, cluster_labels):.4f}")
    print(f'Completeness: {metrics.completeness_score(gt_labels, cluster_labels):.4f}')
    print(f'V-measure: {metrics.v_measure_score(gt_labels, cluster_labels):.4f}')
    print('-----------------------------------------------------------')