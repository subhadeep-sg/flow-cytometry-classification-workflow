import torch
from cnn import ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
import time

st = time.time()

# Need to run only once to obtain directory
# channel1 = get_channel1_unlabelled(path='../dataset/to_predict',
#                                    src='../MasterDataset/',
#                                    verbose=True,
#                                    make_dir=True)

img_dir = '../dataset/to_predict'
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.6733, 0.6733, 0.6733), (0.0598, 0.0598, 0.0598))
])
predict_ds = ImageDataset(img_dir=img_dir, mode='predict', transforms=test_transform)
batch_size = 16
loader = DataLoader(predict_ds, batch_size=batch_size)
print(f'Length of dataset: {len(predict_ds)}')

# Load saved trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '../multicell_classification/saved_models/17102024_1542_88_model.pt'
model = torch.load(model_path, weights_only=False).to(device)
model.eval()

image_list = [path.replace("\\", "/") + '/' + file for path, _, filenames in os.walk(img_dir) for file in filenames]
predictions = []

with torch.no_grad():
    for inputs in loader:
        inputs = inputs.to(device)
        output = model(inputs)
        _, pred = torch.max(output, 1)
        pred = pred.detach().cpu().numpy()
        for pred_element in pred:
            predictions.append(pred_element)

assert len(image_list) == len(predictions), "Image list and prediction list are not same size"

# Now we use multi images for color classification to generate groundtruths
multi_cluster = [image_list[i] for i in range(len(predictions)) if predictions[i] == 2]

single = [image_list[i] for i in range(len(predictions)) if predictions[i] == 1]
print(f'Length of single cell list: {len(single)}')

for i, filename in enumerate(multi_cluster):
    multi_cluster[i] = filename.replace('dataset/to_predict', 'MasterDataset')

print('Length of multi cluster channel 1 image list: ', len(multi_cluster))
print('Adding other channels of the existing predicted multi cluster image...')
multi_copy = []

df = pd.DataFrame(columns=['chan2', 'chan3', 'chan7', 'chan11'])
for filename in multi_cluster:
    chan2 = filename.replace("_chan1_", "_chan2_")
    chan3 = filename.replace("_chan1_", "_chan3_")
    chan7 = filename.replace("_chan1_", "_chan7_")
    chan11 = filename.replace("_chan1_", "_chan11_")

    file2 = os.path.isfile(chan2)
    file3 = os.path.isfile(chan3)
    file7 = os.path.isfile(chan7)
    file11 = os.path.isfile(chan11)

    if file2 and file3 and file7 and file11:
        row = {'chan2': chan2,
               'chan3': chan3,
               'chan7': chan7,
               'chan11': chan11
               }
        df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)
        multi_copy.append(filename)
        multi_copy.append(chan2)
        multi_copy.append(chan3)
        multi_copy.append(chan7)
        multi_copy.append(chan11)

print('Length of new multi-cluster list:', len(multi_copy))

print(df.head())

df.to_csv('../graph_method/filename_list.csv')

print('Time taken: ', time.time() - st)
