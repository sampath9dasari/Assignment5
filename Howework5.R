### Big Data Programming class - Assignment 5
### Susanth Dasari

library(caret)
library(gbm)
library(RANN)
library(ggplot2)

### Using R 3.5.1 (or later) and caret's Animal Scat Data dataset (Hint: data(scat))
data(scat)
str(scat)

### 1
### Set the Species column as the target/outcome and convert it to numeric. (5 points)
target <- ifelse(scat$Species=="bobcat",0,ifelse(scat$Species=="coyote",1,2))
str(target)

### 2
### Remove the Month, Year, Site, Location features. (5 points)
scat$Month <- NULL
scat$Year <- NULL
scat$Site <- NULL
scat$Location <- NULL
str(scat)

### 3
### Check if any values are null. If there are, impute missing values using KNN. (10 points)
sum(is.na(scat))

#Imputing missing values using KNN.Also centering and scaling numerical columns
preProcValues <- preProcess(scat, method = "knnImpute")

scat_processed <- predict(preProcValues, scat)
sum(is.na(scat_processed))


### 4
### Converting every categorical variable to numerical (if needed). (5 points)
str(scat_processed)
# There are no categorical variables remaining


#Converting the dependent variable back to numeric categorical
scat_processed$Species<-as.factor(target)
str(scat_processed)

### 5
### With a seed of 100, 75% training, 25% testing. 
### Build the following models: randomforest, neural net, naive bayes and GBM.
### For these models display 
###   a) model summarization and 
###   b) plot variable of importance, for the predictions (use the prediction set) display 
###   c) confusion matrix (60 points)

########## Splitting Data Using Caret ##############

#Spliting training set into two parts based on outcome: 75% and 25%
set.seed(100)
index <- createDataPartition(scat_processed$Species, p=0.75, list=FALSE)
trainSet <- scat_processed[ index,]
testSet <- scat_processed[-index,]

outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]

######## Training Models Using Caret ############
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')

### Model Statistics for Random Forrest
print(model_rf)
plot(varImp(object=model_rf),main="Random Forrest - Variable Importance")
predictions<-predict.train(object=model_rf,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])


### Model Statistics for Neural Networks
print(model_nnet)
var_nnet <- varImp(object=model_nnet)
barplot(var_nnet$importance[,'Overall'], main="Neural Networks - Variable Importance",
        xlab='Overall', horiz=TRUE, las=2,
        col="darkblue", names.arg=row.names(var_nnet$importance)[])
predictions<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])


### Model Statistics for Naive Bayes
print(model_nb)
plot(varImp(object=model_nb),main="Naive Bayes - Variable Importance")
predictions<-predict.train(object=model_nb,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])


### Model Statistics for GBM
print(model_gbm)
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])

### 6
### For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) 
### create and display a data frame that has the following columns: 
###   ExperimentName, accuracy, kappa. Sort the data frame by accuracy. (15 points)
model_results <- NULL
model_results <- data.frame("ExperimentName" = "Random Forrest",model_rf$results[row.names(model_rf$bestTune),c("Accuracy","Kappa")])
newrow <- data.frame("ExperimentName" = "Neural Networks",model_nnet$results[row.names(model_nnet$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)
newrow <- data.frame("ExperimentName" = "Naive Bayes",model_nb$results[row.names(model_nb$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)
newrow <- data.frame("ExperimentName" = "GBM",model_gbm$results[row.names(model_gbm$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)

# Printing the results in descending order
model_results[order(-model_results$Accuracy),]

### 7
### Tune the GBM model using tune length = 20 and: 
###   a) print the model summary and 
###   b) plot the models. (20 points)

### Using tuneLength ###
model_gbm_tuned <- train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',tuneLength=20)
print(model_gbm_tuned)

# visualize the models
plot(model_gbm_tuned)

### 8
### Using GGplot and gridExtra to plot all variable of importance plots into one single plot. (10 points)
p1 <-ggplot(varImp(object=model_rf)) + ggtitle("Random Forrest")
p2 <-ggplot(varImp(object=model_nb)) + ggtitle("Naive Bayes")
p3 <-ggplot(varImp(object=model_gbm)) + ggtitle("GBM")
var_nnet_df <- data.frame("Column"=row.names(var_nnet$importance)[],varImp(object=model_nnet)$importance)
p4 <- ggplot(var_nnet_df) + geom_col(aes(x=Column,y=Overall)) +  coord_flip() + ggtitle("Neural Networks")

grid.arrange(p1,p2,p3,p4)

### 9
### Which model performs the best? 
### and why do you think this is the case? Can we accurately predict species on this dataset? (10 points)

# After looking at the accuracies of the models, as of now Neural Network is performing the best. 
# But since the accuracy is still 69%, I don’t think we can predict the species accurately. 
# It might because of the low size of data.


### 10
### Graduate Student Questions:
###   a.Using feature selection with rfe in caret and the repeatedcv method: 
###     Find the top 3 predictors and build the same models as in 6 and 8 with the same parameters. (20 points)

#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Scat_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],sizes=3,rfeControl = control)
print(Scat_Pred_Profile)

#Taking only the top 3 predictors
predictors<-c("CN", "d13C", "d15N")


######## Training Models Using Caret ############
model_rf_fs<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet_fs<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_nb_fs<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')
model_gbm_fs<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')


### Tune the GBM model using tune length = 20 and: 
###   a) print the model summary and 
###   b) plot the models.
### Using tuneLength ###

#using tune length
model_gbm_tuned_fs <- train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',tuneLength=20)
print(model_gbm_tuned_fs)

# visualize the models
plot(model_gbm_tuned_fs)

### Using GGplot and gridExtra to plot all variable of importance plots into one single plot.
p1 <-ggplot(varImp(object=model_rf_fs)) + ggtitle("Random Forrest")
p2 <-ggplot(varImp(object=model_nb_fs)) + ggtitle("Naive Bayes")
p3 <-ggplot(varImp(object=model_gbm_fs)) + ggtitle("GBM")
var_nnet_df_fs <- data.frame("Column"=row.names(var_nnet_fs$importance)[],varImp(object=model_nnet_fs)$importance)
p4 <- ggplot(var_nnet_df_fs) + geom_col(aes(x=Column,y=Overall)) +  coord_flip() + ggtitle("Neural Networks")

grid.arrange(p1,p2,p3,p4)

### 10
###   b. Create a dataframe that compares the non-feature selected models ( the same as on 7) 
###     and add the best BEST performing models of each (randomforest, neural net, naive bayes and gbm) 
###     and display the data frame that has the following columns: ExperimentName, accuracy, kappa. 
###     Sort the data frame by accuracy.
newrow <- data.frame("ExperimentName" = "Random Forrest FS",model_rf_fs$results[row.names(model_rf_fs$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)
newrow = data.frame("ExperimentName" = "Neural Networks FS",model_nnet_fs$results[row.names(model_nnet_fs$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)
newrow = data.frame("ExperimentName" = "Naive Bayes FS",model_nb_fs$results[row.names(model_nb_fs$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)
newrow = data.frame("ExperimentName" = "GBM FS",model_gbm_fs$results[row.names(model_gbm_fs$bestTune),c("Accuracy","Kappa")])
model_results <- rbind(model_results,newrow)

# Printing the results in descending order
model_results[order(-model_results$Accuracy),]


### 10
###   c. Which model performs the best? 
###   and why do you think this is the case? 
###   Can we accurately predict species on this dataset? (10 points)

# After doing the feature selection, the best model now is Naive Bayes with Feature selection.
# Since the number of features has greatly reduced and the target variable has only 3 categories,
# It might be the case that Naive Bayes came first.
# But even here, the accuracy is only 73%, so we might still not be able to correctly predict species.
# With a bigger sample, with the same models we might have better results.






