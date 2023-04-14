# not running might be because of missing values 


dataset <- read.csv("C:\\Users\\adamb\\tech\\slu\\spring23\\business-analytics-machine-learning\\titanic-competition\\data\\combined-dataset.csv") # nolint

nrow(dataset)
ncol(dataset)
head(dataset)


trainingsize <- as.integer((1309 - 418) * (1 - 0 / 100))

validationsize <- (1309 - 418) - trainingsize

testingsize <- nrow(dataset) - (1309 - 418)


training <- head(dataset, trainingsize)
validation <- tail(head(dataset, trainingsize + validationsize), validationsize)
testing <- tail(dataset, nrow(dataset) - (trainingsize + validationsize))
if (validationsize == 0) {
    validation <- training
    }

print(nrow(training))
print(nrow(validation))
print(nrow(testing))



install.packages("randomForest")
library(randomForest)

model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = training, ntrees = 500, maxnodes = 64) # nolint

importance(model)
validation$predicted_rf <- predict(model, validation, type = "class")
confusionmatrix <- table(validation$predicted_rf, validation$Survived)
confusionmatrix
accuracy <- (1 - mean(validation$predicted_rf != validation$Survived, na.rm = TRUE)) * 100 # nolint
paste(round(accuracy, 2), "%", sep = "")
predicted_rf <- predict(model, testing, type = "class")
testing <- cbind(testing, predictedRF)
write.csv(testing, "C:\\Users\\adamb\\tech\\slu\\spring23\\business-analytics-machine-learning\\titanic-competition\\random-forest\\combined-dataset-prediction.csv", row.names = FALSE) # nolint
