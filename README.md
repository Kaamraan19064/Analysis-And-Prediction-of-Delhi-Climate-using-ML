# Analysis-And-Prediction-of-Delhi-Climate-using-ML

# Usage 

All source code is in the Code folder. Most of our code is in Python.

It contains three files as follows:

1.Prepocessing.py:It contains all the functions that are required to preprocess the data like to merge the datasets as per gaussian distribution ,one hot encoding ,removing nan values, etc.
it also contains the functions that are required to analyse the data like heat map graph, important features ,etc.
We can run it by just writing python3 Preprocessing.py in our terminal.

2.Regression.py:It contains the functions that are required to predict AQI,future AQI,Variation of AQI in all three seasons,etc using linear regression,SVM,NN,etc.
We can run it by just writing python3 Regression.py in our terminal.

3.Classification.py:It contains the functions that are required to predict extreme weather conditions,Variation of extreme conditions in all three seasons,etc using logistic regression,Random forest, decision tree, etc.
We can run it by just writing python3 classification.py in our terminal.

# Dataset :

We are using two dataset as follows-

A. Dellhi weather dataset 

B. Delhi air quality data

Delhi weather dataset is a time-series hourly data that contains 100990 samples along with 20 features ranging from 1997-2016 Delhi air quality dataset is a time-series daily data that contains 8845 samples along with 13 features ranging from 1995-2015. We merged both the datasets based on date to get a resultant dataset with 57561 samples combining both their features. The dataset was divided into training, validation and test set in the ratio 75:15:10 respectively. Upon close inspection we removed certain unimportant features like ”agency”, etc. We faced issues due to NA values in some columns. So we removed those features which contained more than 15,000 NA values. For the remaining features with NA.

# Methods and Results :

We can see the architecture and results in Docs folder.

https://github.com/Kaamraan19064/Analysis-And-Prediction-of-Delhi-Climate-using-ML/tree/main/Docs

