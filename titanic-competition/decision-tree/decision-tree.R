dataset <- read.csv("C:\\Users\\adamb\\tech\\slu\\spring23\\business-analytics-machine-learning\\titanic-competition\\decision-tree\\combined-dataset.csv") # nolint

if (typeof(dataset$Pclass) == "character") {
    dataset$Pclass <- factor(dataset$Pclass)
    }
if (typeof(dataset$Sex) == "character") {
    dataset$Sex <- factor(dataset$Sex)
    }
if (typeof(dataset$Age) == "character") {
    dataset$Age <- factor(dataset$Age)
    }
if (typeof(dataset$SibSp) == "character") {
    dataset$SibSp <- factor(dataset$SibSp)
    }
if (typeof(dataset$Parch) == "character") {
    dataset$Parch <- factor(dataset$Parch)
    }
if (typeof(dataset$Fare) == "character") {
    dataset$Fare <- factor(dataset$Fare)
    }
if (typeof(dataset$Embarked) == "character") {
    dataset$Embarked <- factor(dataset$Embarked)
    }

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

install.packages("party")

library(party)

model <- ctree(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = training) # nolint: line_length_linter.

model
validation$predicted_dt <- predict(model, validation)
confusionmatrix <- table(validation$predicted_dt, validation$Survived)
confusionmatrix
accuracy <- (1 - mean(validation$predicted_dt != validation$Survived, na.rm = TRUE)) * 100 # nolint: line_length_linter.
paste(round(accuracy, 2), "%", sep = "")
predicted_dt <- predict(model, testing)
plot(model)
testing <- cbind(testing, predicted_dt)
write.csv(testing, "C:\\Users\\adamb\\tech\\slu\\spring23\\business-analytics-machine-learning\\titanic-competition\\decision-tree\\combined-dataset-prediction.csv", row.names = FALSE) # nolint
