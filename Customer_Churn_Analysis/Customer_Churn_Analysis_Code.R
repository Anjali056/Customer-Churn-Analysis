library(plyr)
library(corrplot) #Plot Correlation Matrix
library(dplyr) #Used for Data Manipulation
library(mlbench) 
library(caret) #Classification and Regression Training
library(rpart) #Used for building classification and regression trees 
library(rpart.plot) #Displays trees 
library(randomForest)

churn <- read.csv("D:/Users/gupta/Downloads/Telco_customer_churn.csv")
churn

#shape
dim(churn)
summary(churn)

#checking any null values
sapply(churn, function(x) sum(is.na(x))) 
churn <- churn[complete.cases(churn), ]
sapply(churn, function(x) sum(is.na(x)))
dim(churn)

churnLabel=churn[,27]
#considering col 8 to 26
churn=churn[,8:26]
churn

lapply(churn, unique)

#checking no of unique values in each col
sapply(lapply(churn, unique), length)

#Encoding data
encode=c("SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
         "OnlineSecurity","OnlineBackup","DeviceProtection",
         "TechSupport","StreamingTV","StreamingMovies","PaperlessBilling")
for (i in encode){
  churn[,i] <- as.factor(mapvalues(churn[,i],from=c("No","Yes"),to=c("0", "1")))
}

churn$Gender <- as.factor(mapvalues(churn$Gender,
                                    from=c("Male","Female"),
                                    to=c("0", "1")))


churnLabel <- as.factor(mapvalues(churnLabel,
                                          from=c("No","Yes"),
                                           to=c("0", "1")))

churn

#Plot corelation matrix
numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method="number")



#One-Hot Encoding

dmy <- dummyVars(" ~ .", data = churn, fullRank = T)
dat_transformed <- data.frame(predict(dmy, newdata = churn))
print(dat_transformed)
glimpse(dat_transformed)

#Adding churnlabel COl to Dat_transforemed
df <- data.frame(dat_transformed,churnLabel)
df
dim(df)


#Feature Selection
#Rank Features By Importance
set.seed(7)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(churnLabel~., data=df, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

#Rank Features By Importance
set.seed(7)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(df[,1:23], df[,24], sizes=c(1:23), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))



#Impt Features after Feature Selection
impFeatures <- df[,c("Dependents.1","TenureMonths","TotalCharges","InternetServiceFiber.optic",  
                     "MonthlyCharges","TechSupport.1","ContractOne.year","InternetServiceNo",         
                     "ContractTwo.year","OnlineSecurity.1","PaperlessBilling.1","MultipleLines.1",              
                     "OnlineBackup.1","PaymentMethodElectronic.check","PhoneService.1","StreamingMovies.1",           
                     "Partner.1","StreamingTV.1","SeniorCitizen.1","churnLabel")]
impFeatures
dim(impFeatures)

#splitting the data train 70% and Test=30%
intrain<- createDataPartition(impFeatures$churnLabel,p=0.7,list=FALSE)
set.seed(2017)
training<-impFeatures[intrain,]
testing<- impFeatures[-intrain,]

dim(training) 
dim(testing)

#Logistic Regression
set.seed((50))
LogModel <- glm(churnLabel ~ .,family=binomial(link="logit"),data=training)
fitted.results <- predict(LogModel,newdata=testing,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

#Comparing with test data
misClasificError <- mean(fitted.results != testing$churnLabel)
print(paste('Logistic Regression Accuracy',1-misClasificError))
print("Confusion Matrix for Logistic Regression"); table(testing$churnLabel, fitted.results > 0.5)



#Decision Tree
set.seed(50)
fit <- rpart(churnLabel~., data = training, method = 'class')
rpart.plot(fit, extra = 106)
predict_unseen <-predict(fit, testing, type = 'class')
table_mat <- table(testing$churnLabel, predict_unseen)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))


#Random forest Tree

set.seed(50)
rfModel <- randomForest(churnLabel ~., data = training)
print(rfModel)
pred_rf <- predict(rfModel, testing)
caret::confusionMatrix(pred_rf, testing$churnLabel)


