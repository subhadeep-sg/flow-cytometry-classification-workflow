import shutil
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.feature_selection import VarianceThreshold
from color_classify.feature_vectors import reading_feature_vector
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from sklearn.metrics import f1_score, classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
"""
Attempting to use a small subset of the data to learn and classify 
"""
st = time.time()
fv = reading_feature_vector(file='../graph_method/featurevector.txt', as_arr=True, verbose=False)
df = pd.read_csv('groundtruths.csv', index_col=False)
fv_labels = np.array(df['label'].tolist())
print(fv.shape)
print(fv_labels.shape)

# Preprocessing feature vector
new_fv = StandardScaler().fit_transform(fv)
new_fv = VarianceThreshold(threshold=0.2).fit_transform(new_fv)
# new_fv = PCA(n_components=50, random_state=42).fit_transform(scaled)
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
res_df = pd.DataFrame(columns=['model', 'accuracy', 'f1 score rbc', 'f1 score wbc', 'f1 score wbc_platelet', 'roc-auc'])


def compare_models(model_list, x, y, xval, yval, dataframe):
    for clf in model_list:
        clf.fit(x, y)
        prediction = clf.predict(xval)
        y_prob = clf.predict_proba(xval)
        # print(type(clf).__name__)
        f1 = f1_score(yval, prediction, average=None)
        # print(f'accuracy: {accuracy_score(yval, prediction)}')
        row = pd.DataFrame([{'model': type(clf).__name__,
                             'accuracy': round(accuracy_score(yval, prediction), ndigits=4),
                             'f1 score rbc': round(f1[0], 4),
                             'f1 score wbc': round(f1[1], 4),
                             'f1 score wbc_platelet': round(f1[2], 4),
                             'roc-auc': round(roc_auc_score(y_val, y_prob, multi_class='ovo'), 4)
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
    lgbm.LGBMClassifier(random_state=42, n_estimators=200),
    # GradientBoostingClassifier(n_estimators=200, learning_rate=0.4, random_state=42),
    # AdaBoostClassifier(n_estimators=100, learning_rate=0.4, random_state=42),
]

res_df = compare_models(model_list, x=X_train, y=y_train, xval=X_val, yval=y_val, dataframe=res_df)

print(res_df)
res_df.to_csv('model_comparison.csv', index=False)

# if os.path.exists('model_comparison.xlsx'):
#     os.remove('model_comparison.xlsx')
res_df.to_excel('model_comparison.xlsx', index=False)

print('Time taken:', time.time() - st)
