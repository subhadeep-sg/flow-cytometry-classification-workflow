import cv2
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score, log_loss

st = time.time()
test_export = 'test_exports/ver4ses1_test.csv'
test_df = pd.read_csv(test_export)
test_gt_cluster = os.listdir('../multicell_classification/classify_dataset/test/cluster_revised')
test_gt_non = os.listdir('../multicell_classification/classify_dataset/test/non_cluster_revised')

df = test_df.dropna(axis=0, how='all')
df = df.drop(columns=['Batch_Start'])
# df = df[~df['Filename'].str.contains('Batch', na=False)]
# df = df[~df['Class'].str.contains(', Class', na=False)]
df = df[df['Filename'].str.contains('.png', na=False)]

# filenames = df['Filename'].tolist()
# for file in filenames:
#     if file not in test_gt_cluster and file not in test_gt_non:
#         print(file)
# Some filenames came without the .png extension

# print(df)

df['Filename'] = df['Filename'].apply(lambda x: x + '.png' if '.png' not in x else x)
df['Class'] = df['Class'].apply(lambda x: 'Cluster' if x == ' Cluster' else 'Non-Cluster')
df['gt'] = df['Filename'].apply(lambda x: 'Cluster' if x in test_gt_cluster else 'Non-Cluster')

filename_list = df['Filename'].tolist()
ground_truth = df['gt'].tolist()
predictions = df['Class'].tolist()

gt = [0 if x == 'Non-Cluster' else 1 for x in ground_truth]
pt = [0 if x == 'Non-Cluster' else 1 for x in predictions]

accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average='weighted')
recall = recall_score(ground_truth, predictions, average='weighted')
f1 = f1_score(ground_truth, predictions, average='weighted')
roc = roc_auc_score(gt, pt)
loss = log_loss(gt, pt)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC:", roc)
print("Loss:", loss)

print("\nClassification Report:")
print(classification_report(ground_truth, predictions))
print(len(predictions))


# Add the main directory root to the filenames
cluster_dir = '../multicell_classification/classify_dataset/test/cluster_revised'
non_cluster_dir = '../multicell_classification/classify_dataset/test/non_cluster_revised'
new_filename_list = []
for filename in filename_list:
    if filename in test_gt_cluster:
        new_filename_list.append(os.path.join(cluster_dir, filename))
    elif filename in test_gt_non:
        new_filename_list.append(os.path.join(non_cluster_dir, filename))
    else:
        print('Error! Filename not in either directories!')
        break



def misclassification_saving(preds, gt, image_list):
    relevant_indices = [i for i in range(len(preds)) if preds[i] != gt[i]]

    start = 0
    end = 5
    selected_misses = [image_list[i] for i in relevant_indices[start:end]]
    actual_labels = [gt[i] for i in relevant_indices[start:end]]
    predictions = [preds[i] for i in relevant_indices[start:end]]
    images = [cv2.resize(cv2.imread(img), (224, 224)) for img in selected_misses]

    label_names = ['non cluster', 'cluster']
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(f"Label:{label_names[actual_labels[i]]}, \n Predicted:{label_names[predictions[i]]}")
        ax.axis("off")
    # fig.suptitle(f"{df['Model Name']}")
    plt.tight_layout()
    plt.show()

    # print(len(relevant_indices))
    # print(relevant_indices)
    # print(selected_misses)

    print('---------')


misclassification_saving(pt, gt, new_filename_list)

print('Runtime:', time.time() - st)
