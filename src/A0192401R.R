###################### INITIALIZATION & DATA PREPARATION #######################
set.seed(1101)
setwd("../data")
required_packages = c("class", "rpart", "rpart.plot", "e1071", "ROCR")
# Install any packages that are not already installed
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}
# Load the packages
library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(ROCR)

### Read Data and Declare Categorical Variables
data = read.csv("diabetes-dataset.csv")
data[, c("hypertension", "heart_disease", "diabetes")] = 
  lapply(data[, c("hypertension", "heart_disease", "diabetes")], as.factor)

### Split Data While Maintaining Proportion of Diabetics
pos.data = data[data$diabetes == "1", ]
neg.data = data[data$diabetes == "0", ]
pos.train.index = sample(1:nrow(pos.data))[1:floor(0.8*nrow(pos.data))]
neg.train.index = sample(1:nrow(neg.data))[1:floor(0.8*nrow(neg.data))]
train.data = rbind(pos.data[pos.train.index, ], neg.data[neg.train.index, ])
test.data = rbind(pos.data[-pos.train.index, ], neg.data[-neg.train.index, ])

### Observe Distribution of Each Variable
par(mfrow = c(3, 3))
pie(table(data$gender), main = "Gender")
pie(table(data$hypertension), main = "Hypertension")
pie(table(data$heart_disease), main = "Heart Disease")
pie(table(data$smoking_history), main = "Smoking History")
pie(table(data$diabetes), main = "Diabetes")
hist(data$age, main = "Age")
hist(data$bmi, main = "BMI")
hist(data$HbA1c_level, main = "HbA1c Level")
hist(data$blood_glucose_level, main = "Blood Glucose Level")
par(mfrow = c(1, 1))
# Note: The above histograms and pie charts were visualized in the report
#.      using Tableau software for easier aesthetic control. 

########################## EXPLORATORY DATA ANALYSIS ###########################
### Categorical Data Association
# Contingency Table of Proportion by Categorical Feature
cat_var = c("gender", "hypertension", "heart_disease", "smoking_history")
cont.tables = vector("list", length(cat_var))
names(cont.tables) = cat_var
for (i in seq_along(cat_var)) {
  matrix = table(data[, cat_var[i]], data[, "diabetes"])
  matrix = prop.table(matrix, margin = 1)
  cont.tables[[i]] = matrix
}
cont.tables # <-- Contingency tables for categorical variables w.r.t. diabetes

# Plotting Bar Graphs for Categorical Features
par(mfrow = c(2, 2))
for (i in seq_along(cat_var)) {
  proportions = cont.tables[[i]][, 2]
  max_proportion = max(proportions)
  colors = c("skyblue","lightgreen","orange","pink","yellow","purple","cyan")
  barplot(proportions,
          main = paste("Proportion of Diabetics by", cat_var[i]),
          xlab = cat_var[i],
          ylab = "Proportion with Diabetes",
          ylim = c(0, max_proportion * 1.1),
          col = colors[1:length(proportions)],
          las = ifelse(cat_var[i] == "smoking_history", 2, 0),
          cex.names = 0.8
  )
}
par(mfrow = c(1, 1))

### Quantitative Data Association
# Plotting Box Plots for Quantitative Features
quant_var = c("age", "bmi", "HbA1c_level", "blood_glucose_level")
par(mfrow = c(2, 2))
for (i in seq_along(quant_var)) {
  boxplot(data[, quant_var[i]] ~ data$diabetes,
          xlab = "Diabetes Status",
          ylab = quant_var[i],
          col = c("skyblue", "lightgreen"),
          names = c("Negative", "Positive"),
          main = paste("Distribution of", quant_var[i], "by Diabetes")
  )
}
par(mfrow = c(1, 1))

##################### MODEL BUILDING: K-NEAREST NEIGHBORS ######################
### Standardizing Quantitative Features
train.x = scale(train.data[,c("age","bmi","HbA1c_level","blood_glucose_level")])
train.y = train.data[, "diabetes"]
test.x = scale(test.data[,c("age","bmi","HbA1c_level","blood_glucose_level")])
test.y = test.data[, "diabetes"]

### Choosing K Value
# Compiling Metrics for Each k Value
knn.k = c(1:30)
knn.acc = numeric(0)
knn.tpr = numeric(0)
knn.fpr = numeric(0)
knn.fnr = numeric(0)
knn.prec = numeric(0)
for (k in knn.k) {
  pred.y = knn(train.x, test.x, train.y, k)
  matrix = table(pred.y, test.y)
  matrix = matrix[c(2, 1), c(2, 1)]
  knn.acc = c(knn.acc, sum(diag(matrix))/sum(matrix))
  knn.tpr = c(knn.tpr, matrix[1, 1]/sum(matrix[, 1]))
  knn.fpr = c(knn.fpr, matrix[1, 2]/sum(matrix[, 2]))
  knn.fnr = c(knn.fnr, matrix[2, 1]/sum(matrix[, 1]))
  knn.prec = c(knn.prec, matrix[1, 1]/sum(matrix[1, ]))
  # SORRY THIS LOOP WILL TAKE 2-3 MINUTES TO RUN :(
}
knn.metrics = data.frame(k = knn.k, 
                         Accuracy = knn.acc, 
                         Sensitivity = knn.tpr,
                         Type1_Error = knn.fpr, 
                         Type2_Error = knn.fnr,
                         Precision = knn.prec)
knn.metrics # <-- Compiled metrics for every k value

# Plotting KNN Performance by k Value
plot(knn.metrics$k, knn.metrics$Accuracy,
     main = "KNN Performance by k Value",
     type = "l", col = "blue", lwd = 2,
     xlab = "k", ylab = "Metrics", ylim = c(0, 1))
points(knn.metrics$k, knn.metrics$Accuracy, col = "blue", pch = 19)
lines(knn.metrics$k, knn.metrics$Sensitivity, col = "green", lwd = 2)
points(knn.metrics$k, knn.metrics$Sensitivity, col = "green", pch = 19)
lines(knn.metrics$k, knn.metrics$Type1_Error, col = "red", lwd = 2)
points(knn.metrics$k, knn.metrics$Type1_Error, col = "red", pch = 19)
lines(knn.metrics$k, knn.metrics$Type2_Error, col = "orange", lwd = 2)
points(knn.metrics$k, knn.metrics$Type2_Error, col = "orange", pch = 19)
lines(knn.metrics$k, knn.metrics$Precision, col = "purple", lwd = 2)
points(knn.metrics$k, knn.metrics$Precision, col = "purple", pch = 19)
legend("bottomright",inset = c(0.05, 0.1),
       legend = c("Accuracy",
                  "Precision",
                  "Sensitivity",
                  "Type2Error",
                  "Type1Error"), 
       col = c("blue", "purple", "green", "orange", "red"), lwd = 2)

# Choose k = 3
abline(v = 3, col = "darkgrey", lty = 2)
text(3, 0.5, "k = 3", col = "black")

### Final KNN Model 
knn.model = knn(train.x, test.x, train.y, 3) 

######################### MODEL BULDING: DECISION TREE #########################
### Choosing CP Value
# Compiling Metrics for Each cp Value
dt.cp = c(-10:-1) 
dt.acc = numeric(0)
dt.tpr = numeric(0)
dt.fpr = numeric(0)
dt.fnr = numeric(0)
dt.prec = numeric(0)
for (i in dt.cp) {
  tree = rpart(diabetes ~.,
               method = "class",  
               data = train.data,
               control = rpart.control(cp = 10^i), 
               parms = list(split = 'information'))
  pred.y = predict(tree, newdata = test.data[, c(1:8)], type = "class")
  matrix = table(pred.y, test.data$diabetes)
  matrix = matrix[c(2, 1), c(2, 1)]
  dt.acc = c(dt.acc, sum(diag(matrix))/sum(matrix))
  dt.tpr = c(dt.tpr, matrix[1, 1]/sum(matrix[, 1]))
  dt.fpr = c(dt.fpr, matrix[1, 2]/sum(matrix[, 2]))
  dt.fnr = c(dt.fnr, matrix[2, 1]/sum(matrix[, 1]))
  dt.prec = c(dt.prec, matrix[1, 1]/sum(matrix[1, ]))
}
dt.metrics = data.frame(cp = dt.cp,
                        Accuracy = dt.acc, 
                        Sensitivity = dt.tpr,
                        Type1_Error = dt.fpr, 
                        Type2_Error = dt.fnr,
                        Precision = dt.prec)
dt.metrics # <-- Compiled metrics for every cp value

# Plotting Decision Tree Performance by k Value
plot(dt.metrics$cp, dt.metrics$Accuracy,
     main = "Decision Tree Performance by cp Value",
     type = "l", col = "blue", lwd = 2,
     xlab = "log10(cp)", ylab = "Metrics", ylim = c(0, 1))
points(dt.metrics$cp, dt.metrics$Accuracy, col = "blue", pch = 19)
lines(dt.metrics$cp, dt.metrics$Sensitivity, col = "green", lwd = 2)
points(dt.metrics$cp, dt.metrics$Sensitivity, col = "green", pch = 19)
lines(dt.metrics$cp, dt.metrics$Type1_Error, col = "red", lwd = 2)
points(dt.metrics$cp, dt.metrics$Type1_Error, col = "red", pch = 19)
lines(dt.metrics$cp, dt.metrics$Type2_Error, col = "orange", lwd = 2)
points(dt.metrics$cp, dt.metrics$Type2_Error, col = "orange", pch = 19)
lines(dt.metrics$cp, dt.metrics$Precision, col = "purple", lwd = 2)
points(dt.metrics$cp, dt.metrics$Precision, col = "purple", pch = 19)
legend("bottomright",inset = c(0.05, 0.1),
       legend = c("Accuracy",
                  "Precision",
                  "Sensitivity",
                  "Type2Error",
                  "Type1Error"), 
       col = c("blue", "purple", "green", "orange", "red"), lwd = 2)

# Choose cp = 10^(-4)
abline(v = -4, col = "darkgrey", lty = 2)
text(-4, 0.5, "cp = 10^-4", col = "black")

### Final Decision Tree Model
dt.model = rpart(diabetes ~.,
                 method = "class",  
                 data = train.data,
                 control = rpart.control(cp = 10^(-4)), 
                 parms = list(split = 'information'))
dt.diagram = rpart.plot(dt.model, type = 2) # too many branches to visualize

######################## MODEL BUILDING: NAIVE BAYES ###########################
nb.model = naiveBayes(diabetes ~ 
                        gender + 
                        hypertension + 
                        heart_disease + 
                        smoking_history, 
                      train.data)

##################### MODEL BUILDING: LOGISTIC REGRESSION ######################
lr.model = glm(diabetes ~., data = data, family = binomial(link ="logit"))
summary(lr.model)
# Note: The equation is illustrated in the report using LaTeX.

######################## MODEL EVALUTION AND COMPARISON ########################
### Prediction Using Test Data
knn.predclass = knn.model
knn.predprob = as.numeric(paste(knn.model))
# Note: Setting prob=TRUE for knn() returns inconsistent probabilities that 
#.      do not match the predicted class when delta = 0.5, resulting in a 
#.      ROC curve that falls below the diagonal. To avoid this unintended 
#.      behavior, the binary output of the  model will be used to plot the 
#.      ROC curve instead. This approach was cleared with Ms Daisy :)

dt.predclass = predict(dt.model, newdata = test.data[, c(1:8)], type = "class")
dt.predprob = predict(dt.model, newdata = test.data[, c(1:8)], type = "prob")

nb.predclass = predict(nb.model, newdata = test.data[, cat_var], "class")
nb.predprob = predict(nb.model, newdata = test.data[, cat_var], "raw")
  
lr.predprob = predict(lr.model, newdata = test.data[, c(1:8)], type='response')
lr.predclass = ifelse(lr.predprob >= 0.5, 1, 0)

### Compiling Metrics for Each Model
pred_list = list("K Nearest Neighbours" = knn.predclass, 
                 "Decision Tree" = dt.predclass, 
                 "Naive Bayes" = nb.predclass, 
                 "Logistic Regression" = lr.predclass)

metrics = data.frame(Model = character(),
                     Accuracy = numeric(),
                     Sensitivity = numeric(),
                     Type1Error = numeric(),
                     Type2Error = numeric(),
                     Precision = numeric())

for (model in names(pred_list)) {
  pred.y = pred_list[[model]]
  matrix = table(pred.y, test.data$diabetes)
  matrix = matrix[c(2, 1), c(2, 1)]
  acc = round(sum(diag(matrix))/sum(matrix), 5)
  tpr = round(matrix[1, 1]/sum(matrix[, 1]), 5)
  fpr = round(matrix[1, 2]/sum(matrix[, 2]), 5)
  fnr = round(matrix[2, 1]/sum(matrix[, 1]), 5)
  prec = round(matrix[1, 1]/sum(matrix[1, ]), 5)
  metrics[nrow(metrics) + 1, ] = c(model, acc, tpr, fpr, fnr, prec)
}
metrics # <-- Compiled metrics for each model, visualized in report using Excel

# Note: There are negligible discrepancies observed in the metrics for the
#.      final knn model compared to the knn.metrics table at k = 3. This could
#.      be attributed to how the algorithm handles tie-breaking.

### Receiving Operating Characteristic & Area Under Curve
knn.format = prediction(knn.predprob, test.data$diabetes)
dt.format = prediction(dt.predprob[, 2], test.data$diabetes)
nb.format = prediction(nb.predprob[, 2], test.data$diabetes)
lr.format = prediction(lr.predprob, test.data$diabetes)

knn.perf = performance(knn.format, "tpr", "fpr")
dt.perf = performance(dt.format, "tpr", "fpr")
nb.perf = performance(nb.format, "tpr", "fpr")
lr.perf = performance(lr.format, "tpr", "fpr")

plot(knn.perf, lwd = 2, col = 'red', main = "ROC Curves for Various Models",
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")
plot(dt.perf, lwd = 2, col = 'blue', add = TRUE)
plot(nb.perf, lwd = 2, col = 'green', add = TRUE)
plot(lr.perf, lwd = 2, col = 'purple', add = TRUE)
abline (a=0, b=1, col ="grey", lty =3)

knn.auc = round(performance(knn.format, "auc")@y.values[[1]], 3)
dt.auc = round(performance(dt.format, "auc")@y.values[[1]], 3)
nb.auc = round(performance(nb.format, "auc")@y.values[[1]], 3)
lr.auc = round(performance(lr.format, "auc")@y.values[[1]], 3)

legend("bottomright", inset = c(0.05, 0.05), title = "Model AUC Values",
       legend = c(paste("K-Nearest Neighbours:", knn.auc),
                  paste("Decision Tree:", dt.auc),
                  paste("Naive Bayes:", nb.auc),
                  paste("Logistic Regression:", lr.auc)),
       col = c("red", "blue", "green","purple"), lty = 1, lwd = 2, cex = 1.3)

################################ END OF REPORT #################################