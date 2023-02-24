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

nrow(training)
nrow(validation)
nrow(testing)

install.packages("e1071")
library(e1071)

model <- naiveBayes(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = training) # nolint

model

validation$predicted_nb <- predict(model, validation)
confusionmatrix <- table(validation$predicted_nb, validation$Survived)
confusionmatrix

accuracy <- (1 - mean(validation$predicted_nb != validation$Survived, na.rm = TRUE)) * 100 # nolint

paste(round(accuracy, 2), "%", sep = "")

predicted_nb <- predict(model, testing)
testing <- cbind(testing, predicted_nb)
write.csv(testing, "C:\\Users\\adamb\\tech\\slu\\spring23\\business-analytics-machine-learning\\titanic-competition\\naive-bayes\\combined-dataset-prediction.csv", row.names = FALSE) # nolint
