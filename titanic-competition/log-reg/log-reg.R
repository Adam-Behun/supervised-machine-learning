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

model <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data=training,family="binomial") # nolint

summary(model)

validation$predicted_lr <- predict(model, validation, type = "response")
predictedclass <- ifelse(validation$predicted_lr > 0.5, 1, 0)

confusionmatrix <- table(factor(predictedclass, levels = 0:1), factor(validation$Survived, levels = 0:1)) # nolint

confusionmatrix
accuracy <- (1 - mean(predictedclass != validation$Survived, na.rm = TRUE)) * 100 # nolint
paste(round(accuracy, 2), "%", sep = "")
predicted_lr <- predict(model, testing, type = "response")
predicted_lr <- ifelse(predicted_lr > 0.5, 1, 0)
testing <- cbind(testing, predicted_lr)
write.csv(testing, "C:\\Users\\adamb\\tech\\slu\\spring23\\business-analytics-machine-learning\\titanic-competition\\log-reg\\combined-dataset-prediction.csv", row.names = FALSE) # nolint
