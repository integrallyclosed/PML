Predicting activity class using accelerometer data
========================================================

In the following analysis I use data from accelerometers of fitness devices to predict the activity class. Specifically, I will use accelerometer data from the belt, forearm, arm, and dumbell of 6 participants. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

## Download data

The data for this exercise comes from one of the Human activity recognition datasets found at http://groupware.les.inf.puc-rio.br/har. Let's first download the data as the file pml-training.csv.


```r
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    destfile = "pml-training.csv", method = "curl")
```


## Load the data

Next load the dataset into an R data frame


```r
pml <- read.csv("pml-training.csv")
```



## Preliminary analysis
There are 160 columns in the dataset.

```r
print(dim(pml)[2])
```

```
## [1] 160
```


There are 19622 records for which data is available as can be seen from the number of rows.

```r
print(dim(pml)[1])
```

```
## [1] 19622
```


The goal of this exercise is to build a prediction model to predict the classe variable. The barbell lifts activity has been codified into factor levels A, B, C, D, E.

```r
print(levels(pml$classe))
```

```
## [1] "A" "B" "C" "D" "E"
```


Since the problem is a classification problem where the predicted variable has five possible values, a multi class classification algorithm needs to be employed. I have chosen to use a random forests classifier. 

At this point, a choice needs to be made for what predictor variables should be chosen for the algorithm. If we look at the actual entries in the dataset, it can be seen that a large number of columns have most of their entries either NA or blank. I wanted to remove these columns from the set of predictor variables. 

To find such columns, we can run:

```r
columns_without_na_or_blanks <- (sapply(pml, function(x) {
    sum(is.na(x) | x == "")
}) == 0)
print(sum(columns_without_na_or_blanks))
```

```
## [1] 60
```

Thus, there are only 60 columns where there are no NA or blank entries.
Let's remove these columns from the dataset and store it in a new dataframe.

```r
pml_subset <- pml[, columns_without_na_or_blanks]
```


We can eliminate some more columns before finalizing our predictor set. The first seven columns of the dataset seem to have very little predictive value. 

```r
print(names(pml_subset)[1:7])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

The first column is just a serial number of the observation. The second is the user name of the participant. The next three columns are time stamps. The next two columns have information about the activity window which is related to the measurement time. Because of the nature of these columns, I don't expect any of these to contribute to the final activity classe prediction. Let's remove these columns from the dataset as well. The final dataset has 53 columns.

```r
pml_subset <- pml_subset[, 8:60]
print(dim(pml_subset))
```

```
## [1] 19622    53
```


## Building a predictive model
We will use the caret library to build a predictive model. We first create a 75-25 partition of the dataset into train and test sets.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(1)
inTrain <- createDataPartition(y = pml_subset$classe, p = 0.75, list = F)
pml_training <- pml_subset[inTrain, ]
pml_testing <- pml_subset[-inTrain, ]
```


For some exploratory data analysis, let us observe the distribution of the classe variable in the training and test sets.

```r
barplot(prop.table(table(pml_training$classe)), main = "Distribution of classe variable in training data", 
    ylab = "Fraction")
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-111.png) 

```r
barplot(prop.table(table(pml_testing$classe)), main = "Distribution of classe variable in testing data", 
    ylab = "Fraction")
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-112.png) 


Now we train a random forest model on the training set with classe variable as the target variable and all the remaining variables as predictors. To compute the out of sample error we will do a 5-fold cross validation. With the default value for random forests, the model takes a long time to train. In order to reduce training time, I changed ntree parameter to 100 instead of the default 500. This significantly reduces training time without reducing precision appreciably. Further, since the machine on which I built the model has 8 cores, I used a parallelized approach to train the model using the library doMC. Using these techniques I was able to reduce the training time from 40 minutes with the default values to under a minute.

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(cores = 8)
set.seed(1)
cvControl <- trainControl(method = "cv", number = 5)
modelFit <- train(classe ~ ., data = pml_training, method = "rf", ntree = 100, 
    trControl = cvControl)
```


Let's print the cross validation results.

```r
print(modelFit)
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 11774, 11774, 11776, 11773, 11775 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.003   
##   30    1         1      0.002        0.003   
##   50    1         1      0.002        0.002   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


Since the accuracy in the cross validation runs is over 0.99, it is getting rounded off to one. We can plot the results of the cross validation as follows to see the actual accuracy numbers.

```r
plot(modelFit)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14.png) 


Finally, we will make predictions on the test data and look at the confusion matrix of the predictions compared with the true classe labels.

```r
predictions <- predict(modelFit, newdata = pml_testing)
confusionMatrix(predictions, pml_testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1391    5    0    0    0
##          B    2  942    3    0    0
##          C    1    2  850    5    1
##          D    0    0    2  798    3
##          E    1    0    0    1  897
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.992, 0.997)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.993    0.994    0.993    0.996
## Specificity             0.999    0.999    0.998    0.999    1.000
## Pos Pred Value          0.996    0.995    0.990    0.994    0.998
## Neg Pred Value          0.999    0.998    0.999    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.163    0.183
## Detection Prevalence    0.285    0.193    0.175    0.164    0.183
## Balanced Accuracy       0.998    0.996    0.996    0.996    0.998
```


Thus, we estimate the out of the sample error to be about 0.5%.

We can look at the most important predictor variables as follows.

```r
varImp(modelFit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt              100.0
## pitch_forearm           61.6
## yaw_belt                56.5
## magnet_dumbbell_y       46.1
## pitch_belt              44.5
## roll_forearm            44.0
## magnet_dumbbell_z       43.2
## accel_dumbbell_y        22.5
## accel_forearm_x         17.3
## roll_dumbbell           16.7
## magnet_dumbbell_x       16.3
## magnet_belt_z           15.4
## accel_dumbbell_z        14.3
## magnet_belt_y           14.0
## total_accel_dumbbell    14.0
## magnet_forearm_z        13.3
## gyros_belt_z            12.6
## accel_belt_z            12.1
## magnet_belt_x           11.3
## yaw_arm                 10.8
```


We can visualize the most important variables using the following plot

```r
plot(varImp(modelFit))
```

![plot of chunk unnamed-chunk-17](figure/unnamed-chunk-17.png) 

