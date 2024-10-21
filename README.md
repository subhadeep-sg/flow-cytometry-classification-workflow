# flow cytometry classification workflow
 Code for dataset generation and EDA of flow cytometry images

#### Folders:

MasterDataset: Contains all the images in the dataset unorganized.

prepare_data_source: Contains python files to manually label data and create the data directory

dataset: Contains the Channel 1 files organized into 


Steps:
1. Using the MasterDataset folder, we manually label a few images as multi-clusters or single and particle images.


#### Part 1


#### Part 2

Using either the predictions of the model trained in part 1 or a pre-existing multi-cluster dataset available 
in prepare_data_source/revised_clusters/multi we generate groundtruths for all images using
color_classification/get_gt.py

Using the pre-existing dataset using
color_classification/create_multi_groundtruth.py we generate multi_data_list.csv
This csv file is passed to get_gt.py to generate groundtruths.csv

Verify the quality of the groundtruths generated using color_classification/verify_groundtruths.py

With the groundtruths, train the model using color_classification/rgb_classify.py
If save is set to True, the results from cross validation are saved in color_classification/cross_validation_results.csv

For testing, copy the model path stored in color_classification/saved_rgb_model and paste into model load in 
color_classification/rgb_test.py.
This results in generating test_set_results.csv as well as color_classification/p2_misclassifications
which contains plots of images that were misclassified by the model.





