import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgbm
from skimage.feature import hog
import time

st = time.time()
# Define your dataset path
data_path = './classify_dataset'
categories = ['cluster_revised', 'non_cluster_revised']
resize_dim = (64, 64)


# Choose a feature extractor
def extract_features(img):
    img = cv2.resize(img, resize_dim)
    return hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


data, labels = [], []
for category in categories:
    label = 0 if category == 'cluster_revised' else 1
    folder = os.path.join(data_path, 'train', category)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            features = extract_features(img)
            data.append(features)
            labels.append(label)

X = np.array(data)
y = np.array(labels)

# -------- Prepare Test Set --------
test_data, test_labels = [], []
for category in categories:
    label = 0 if category == 'cluster_revised' else 1
    folder = os.path.join(data_path, 'test', category)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            features = extract_features(img)
            test_data.append(features)
            test_labels.append(label)

X_test = np.array(test_data)
y_test = np.array(test_labels)


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_stratify = np.concatenate([y_train, y_val])  # For class weight calculation


# Model comparison function (your code, cleaned up)
# def compare_models(model_list, x, y, xval, yval, dataframe):
#     for clf in model_list:
#         clf.fit(x, y)
#         prediction = clf.predict(xval)
#         y_prob = clf.predict_proba(xval)
#         f1 = f1_score(yval, prediction, average='macro')
#         precision = precision_score(yval, prediction, average='macro')
#         recall = recall_score(yval, prediction, average='macro')
#         loss = log_loss(yval, y_prob)
#
#         row = pd.DataFrame([{
#             'model': type(clf).__name__,
#             'Accuracy': round(accuracy_score(yval, prediction), 4),
#             'F1-score': round(f1, 4),
#             'ROC-AUC': round(roc_auc_score(yval, y_prob[:, 1]), 4),
#             'Precision': round(precision, 4),
#             'Recall': round(recall, 4),
#             'Loss': round(loss, 4)
#         }])
#         dataframe = pd.concat([dataframe, row], ignore_index=True)
#     return dataframe

def compare_models(model_list, x_train, y_train, x_val, y_val, x_test, y_test, dataframe):
    for clf in model_list:
        clf.fit(x_train, y_train)

        for split_name, x_eval, y_eval in [('val', x_val, y_val), ('test', x_test, y_test)]:
            prediction = clf.predict(x_eval)
            y_prob = clf.predict_proba(x_eval)

            f1 = f1_score(y_eval, prediction, average='macro')
            precision = precision_score(y_eval, prediction, average='macro')
            recall = recall_score(y_eval, prediction, average='macro')
            loss = log_loss(y_eval, y_prob)
            roc = roc_auc_score(y_eval, y_prob[:, 1])

            row = pd.DataFrame([{
                'model': type(clf).__name__,
                'dataset': split_name,
                'accuracy': round(accuracy_score(y_eval, prediction), 4),
                'f1 (macro)': round(f1, 4),
                'roc_auc': round(roc, 4),
                'precision (macro)': round(precision, 4),
                'recall (macro)': round(recall, 4),
                'cross_entropy_loss': round(loss, 4),

            }])

            dataframe = pd.concat([dataframe, row], ignore_index=True)

    return dataframe


# Compute class weights
class_weight = list(compute_class_weight(class_weight='balanced', classes=np.unique(y_stratify), y=y_stratify))
print(f'Class weights: {class_weight}')

# Define models
model_list = [
    SVC(C=2, probability=True, random_state=42),
    KNeighborsClassifier(n_neighbors=13),
    RandomForestClassifier(n_estimators=400, random_state=42),
    xgb.XGBClassifier(objective='binary:logistic',
                      learning_rate=0.02, n_estimators=200, random_state=42),
    lgbm.LGBMClassifier(random_state=42, n_estimators=200),
]

# Run model comparisons
# res_df = pd.DataFrame()
# res_df = compare_models(model_list, x=X_train, y=y_train, xval=X_val, yval=y_val, dataframe=res_df)
# print(res_df)
# res_df.to_csv('mlclassifier_comparison.csv', index=False)

res_df = pd.DataFrame()
res_df = compare_models(model_list, X_train, y_train, X_val, y_val, X_test, y_test, res_df)
res_df.to_csv('model_comparison_full.csv', index=False)
print(res_df)

print('Time taken:', time.time()-st)