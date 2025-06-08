import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight
from skimage.feature import hog
import cv2
import os
from color_classification.color_classify import get_feature_vector, saving_feature_vector, reading_feature_vector

st = time.time()
df = pd.read_csv('../color_classification/groundtruths.csv')
print(df.columns)
print(df.label.value_counts())
# feature_vector = get_feature_vector(df)
feature_vector = reading_feature_vector('featurevector.txt', as_arr=True)

fv_labels = np.array(df['label'].tolist())

resize_dim = (64, 64)  # or your preferred size


def extract_hog_features_from_row(row):
    hog_features = []
    for channel in ['chan2', 'chan3', 'chan7', 'chan11']:  # adjust if your columns are named differently
        img_path = os.path.join('../color_classification/channel_images', row[channel])  # update path if needed
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, resize_dim)
            features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            hog_features.extend(features)
        else:
            # If the image is missing or unreadable, fill with zeros of expected length
            hog_features.extend([0] * (resize_dim[0] // 8 * resize_dim[1] // 8 * 4))  # rough estimate
    return np.array(hog_features)


# Apply feature extraction to the entire dataframe
hog_feature_vector = np.array(df.apply(extract_hog_features_from_row, axis=1).tolist())

# Preprocessing feature vector
# new_fv = StandardScaler().fit_transform(hog_feature_vector)#feature_vector)
# new_fv = VarianceThreshold(threshold=0.2).fit_transform(new_fv)
# new_fv = PCA(n_components=50, random_state=42).fit_transform(scaled)

new_fv = hog_feature_vector
print(f'Post processing: {new_fv.shape}')

# Getting train, test, valid splits.
# Starting with very low training data
test_size = 0.3
valid_split = 0.2
train_size = 1 - test_size
valid_size = valid_split / train_size

le = LabelEncoder().fit(fv_labels)
print(le.classes_)
y_stratify = LabelEncoder().fit_transform(fv_labels)

X_train, X_test, y_train, y_test = train_test_split(new_fv, y_stratify, test_size=test_size, random_state=42,
                                                    stratify=y_stratify, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valid_size, random_state=42,
                                                  stratify=y_train, shuffle=True)
#
print(f'Train set contains {X_train.shape[0]} rows')
print(f'Valid set contains {X_val.shape[0]} rows')

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = SVC(random_state=42)
# model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, random_state=42, loss='exponential')
# model = xgb.XGBClassifier(objective='multi:softprob', learning_rate=0.1, n_estimators=100)
# model.fit(X_train, y_train)
# pred = model.predict(X_val)

# Organizing a dataframe to store results
res_df = pd.DataFrame()  # columns=['model', 'accuracy', 'f1 score rbc', 'f1 score wbc', 'f1 score wbc_platelet', 'roc-auc'])


def validation_score_comparison(model_list, x, y, xval, yval, dataframe):
    for clf in model_list:
        clf.fit(x, y)
        prediction = clf.predict(xval)
        y_prob = clf.predict_proba(xval)
        # print(type(clf).__name__)
        f1 = f1_score(yval, prediction, average=None)
        # print(f'accuracy: {accuracy_score(yval, prediction)}')
        row = pd.DataFrame([{'model': type(clf).__name__,
                             'accuracy': round(accuracy_score(yval, prediction) * 100, ndigits=4),
                             # 'f1 score rbc': round(f1[0], 4),
                             # 'f1 score wbc': round(f1[1], 4),
                             # 'f1 score wbc_platelet': round(f1[2], 4),
                             'f1': round(f1_score(y_val, prediction, average='weighted'), 4),
                             'roc-auc': round(roc_auc_score(y_val, y_prob, multi_class='ovo'), 4),
                             'precision': round(precision_score(y_val, prediction, average='weighted'), 4),
                             'recall': round(recall_score(y_val, prediction, average='weighted'), 4),
                             'loss': round(log_loss(y_val, y_prob), 4)

                             }])
        dataframe = pd.concat([dataframe, row], ignore_index=True)
        # print(f'roc-auc: {roc_auc_score(yval, prediction, multi_class="ovr")}')
        # print(classification_report(yval, prediction))
        # print('------------------------')
    return dataframe


class_weight = list(compute_class_weight(class_weight='balanced', classes=np.unique(y_stratify), y=y_stratify))
print(f'Class weights: {class_weight}')
model_list = [
    SVC(C=2, random_state=42, probability=True),
    KNeighborsClassifier(n_neighbors=13),
    RandomForestClassifier(n_estimators=400, random_state=42),
    xgb.XGBClassifier(objective='multi:softprob', sample_weight=class_weight, learning_rate=0.02,
                      n_estimators=200),
    lgbm.LGBMClassifier(random_state=42, n_estimators=200, verbose=0),
    # GradientBoostingClassifier(n_estimators=200, learning_rate=0.4, random_state=42),
    # AdaBoostClassifier(n_estimators=100, learning_rate=0.4, random_state=42),
]

res_df = validation_score_comparison(model_list, x=X_train, y=y_train, xval=X_val, yval=y_val, dataframe=res_df)

res_df.to_csv('mlclassifiers_validation.csv', index=False)
res_df.to_excel('mlclassifiers_validation.xlsx', index=False)

test_df = pd.DataFrame()


def test_comparison(model_list, x, y, xtest, ytest, dataframe):
    for clf in model_list:
        clf.fit(x, y)
        prediction = clf.predict(xtest)
        y_prob = clf.predict_proba(xtest)
        row = pd.DataFrame([{'model': type(clf).__name__,
                             'accuracy': round(accuracy_score(ytest, prediction) * 100, 4),
                             'f1': round(f1_score(ytest, prediction, average='weighted'), 4),
                             'roc-auc': round(roc_auc_score(ytest, y_prob, multi_class='ovo'), 4),
                             'precision': round(precision_score(ytest, prediction, average='weighted'), 4),
                             'recall': round(recall_score(ytest, prediction, average='weighted'), 4),
                             'loss': round(log_loss(ytest, y_prob), 4)
                             }])
        dataframe = pd.concat([dataframe, row], ignore_index=True)
        # print(f'roc-auc: {roc_auc_score(yval, prediction, multi_class="ovr")}')
        # print(classification_report(yval, prediction))
        # print('------------------------')
    return dataframe


test_df = test_comparison(model_list, x=X_train, y=y_train, xtest=X_test, ytest=y_test, dataframe=test_df)
print(test_df)
test_df.to_csv('hog_mlclassifiers_test.csv', index=False)
# test_df.to_excel('hog_mlclassifiers_test.xlsx', index=False)

print('Time taken:', time.time() - st)
