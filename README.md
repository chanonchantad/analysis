# analysis
python files related to running analysis of the results of AI trained algorithms (Mostly supervised training CNNs).

This repository is a collection of code written by me and partly by CAIDM director Dr. Peter Chang (https://github.com/peterchang77/). It's main purpose is for me to keep a reference of the various techniques we use for our AI training results analysis in many medical deep learning projects.

### heatmap.py
When our team wants to do further analysis on where the model focuses on in the data to make it's decision. This will highlight and color the areas that the model is interested in and provides some insight on the important parts of the image that influences its decisions.

### roc_auc.py
When writing a paper for our AI project, a standard metric that usually gets reported is the AUC. This serves to show how well our model performed by providing a graph visualization of the false positive rate (fpr) and true positibe rate (tpr) accross all possible thresholds for a binary classifier.