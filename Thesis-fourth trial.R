# Load required libraries
library(tidyverse)
library(caret)
library(e1071)         # For SVM
library(randomForest)
library(gbm)           # Gradient Boosting
library(xgboost)       # XGBoost
library(MLmetrics)
library(PerformanceAnalytics)
library(nortest)
library(car)
library(factoextra)
library(ggplot2)
library(tidyr)

# Set seed using your student ID
set.seed(311441)

# Load the dataset
data <- read.csv("/Users/bentebeelen/documents/shopping_behavior_updated.csv")
summary(data)

# Convert target and relevant predictors to factors
data$Discount.Applied <- as.factor(data$Discount.Applied)
data$Promo.Code.Used <- as.factor(data$Promo.Code.Used)
data$Shipping.Type <- as.factor(data$Shipping.Type)
data$Frequency.of.Purchases <- as.factor(data$Frequency.of.Purchases)
data$Payment.Method <- as.factor(data$Payment.Method)
data$Gender_numeric <- ifelse(data$Gender == "Male", 1, 
                              ifelse(data$Gender == "Female", 0, NA))

# Define the outcome and predictors
data_model <- data %>%
  select(Purchase.Amount..USD., Discount.Applied, Promo.Code.Used, Shipping.Type, Previous.Purchases, Gender_numeric, Age, Frequency.of.Purchases, Payment.Method)

# Split the dataset
trainIndex <- createDataPartition(data_model$Purchase.Amount..USD., p = 0.6, list = FALSE)
train <- data_model[trainIndex, ]
temp <- data_model[-trainIndex, ]
validIndex <- createDataPartition(temp$Purchase.Amount..USD., p = 0.5, list = FALSE)
valid <- temp[validIndex, ]
test <- temp[-validIndex, ]

# ---- LINEAR REGRESSION ----
lm_model <- lm(Purchase.Amount..USD. ~ ., data = train)
lm_preds <- predict(lm_model, newdata = test)
lm_results <- evaluate_model(test$Purchase.Amount..USD., lm_preds)
plot(lm_preds, test$Purchase.Amount..USD. - lm_preds, 
     main = "Residual Plot: Linear Regression", 
     xlab = "Predicted", ylab = "Residuals", col = "darkred", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- SVM ----
svm_model <- svm(Purchase.Amount..USD. ~ ., data = train)
svm_preds <- predict(svm_model, newdata = test)
svm_results <- evaluate_model(test$Purchase.Amount..USD., svm_preds)
plot(svm_preds, test$Purchase.Amount..USD. - svm_preds, 
     main = "Residual Plot: SVM", 
     xlab = "Predicted", ylab = "Residuals", col = "darkgreen", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- RANDOM FOREST ----
rf_model <- randomForest(Purchase.Amount..USD. ~ ., data = train)
rf_preds <- predict(rf_model, newdata = test)
rf_results <- evaluate_model(test$Purchase.Amount..USD., rf_preds)
plot(rf_preds, test$Purchase.Amount..USD. - rf_preds, 
     main = "Residual Plot: Random Forest", 
     xlab = "Predicted", ylab = "Residuals", col = "darkblue", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- GRADIENT BOOSTING ----
gbm_model <- gbm(Purchase.Amount..USD. ~ ., 
                 data = train, 
                 distribution = "gaussian", 
                 n.trees = 100)
gbm_preds <- predict(gbm_model, newdata = test, n.trees = 100)
gbm_results <- evaluate_model(test$Purchase.Amount..USD., gbm_preds)
plot(gbm_preds, test$Purchase.Amount..USD. - gbm_preds, 
     main = "Residual Plot: GBM", 
     xlab = "Predicted", ylab = "Residuals", col = "purple", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- XGBOOST ----
train_matrix <- model.matrix(Purchase.Amount..USD. ~ . -1, data = train)
test_matrix <- model.matrix(Purchase.Amount..USD. ~ . -1, data = test)
xgb_train <- xgb.DMatrix(data = train_matrix, label = train$Purchase.Amount..USD.)
xgb_test <- xgb.DMatrix(data = test_matrix, label = test$Purchase.Amount..USD.)

xgb_model <- xgboost(data = xgb_train, objective = "reg:squarederror", nrounds = 100, verbose = 0)
xgb_preds <- predict(xgb_model, newdata = xgb_test)
xgb_results <- evaluate_model(test$Purchase.Amount..USD., xgb_preds)

plot(xgb_preds, test$Purchase.Amount..USD. - xgb_preds, 
     main = "Residual Plot: XGBoost", 
     xlab = "Predicted", ylab = "Residuals", col = "orange", pch = 20)
abline(h = 0, col = "blue", lty = 2)
par(mfrow = c(1, 1))

# --- Output results ---
cat("=== Linear Regression ===\n")
print(lm_results)

cat("\n=== SVM ===\n")
print(svm_results)

cat("\n=== Random Forest ===\n")
print(rf_results)

cat("\n=== GBM ===\n")
print(gbm_results)

cat("\n=== XGBoost ===\n")
print(xgb_results)

# --- Correlation Plot ---
# Create working copy of correlation data
corr_data <- data_model[, c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases")]

# Ensure all are numeric and remove rows with NA
corr_data <- as.data.frame(sapply(corr_data, as.numeric))
corr_data <- na.omit(corr_data)
names(corr_data)[names(corr_data) == "Purchase.Amount..USD."] <- "Pn"

# Plot updated correlation matrix
chart.Correlation(corr_data, histogram = TRUE, pch = 19)

# --- Univariate Normality Tests ---
cat("\n=== Univariate Normality Tests ===\n")
num_vars <- sapply(df, is.numeric)
for (var in names(df)[num_vars]) {
  cat("\nVariable:", var, "\n")
  print(shapiro.test(df[[var]]))
  print(lillie.test(df[[var]]))
  print(ad.test(df[[var]]))
}

# --- Influence Diagnostics ---
cat("\n=== Influence Diagnostics ===\n")
influencePlot(lm_model, id.method = "identify", main = "Cook's D Bar Plot", sub = "Threshold: 4/n")
par(mfrow = c(2, 3))
dfb <- dfbetas(lm_model)
for (var in colnames(dfb)) {
  plot(dfb[, var], type = "h", main = paste("Influence Diagnostics for", var),
       xlab = "Observation", ylab = "DFBETAS", col = "blue")
  abline(h = c(-1, 1) * 2 / sqrt(nrow(train)), col = "red", lty = 2)
}
plot(dffits(lm_model), type = "h", col = "blue", main = "Influence Diagnostics for Purchase Amount",
     xlab = "Observation", ylab = "DFFITS")
abline(h = c(-1, 1) * 2 * sqrt(length(coef(lm_model)) / nrow(train)), col = "red", lty = 2)
par(mfrow = c(1, 1))

# --- Clustering and Cluster Boxplots ---
cat("\n=== Clustering and Cluster Boxplots ===\n")
cluster_vars <- data_model[, c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases")]

# Ensure all columns are numeric and remove any rows with NA
cluster_vars <- as.data.frame(sapply(cluster_vars, as.numeric))
cluster_vars <- na.omit(cluster_vars)

# Now scale
cluster_vars_scaled <- scale(cluster_vars)

# K-means clustering
kmeans_result <- kmeans(cluster_vars_scaled, centers = 3, nstart = 25)
data_model$cluster <- as.factor(kmeans_result$cluster)

# PCA for visualization
pca_res <- prcomp(cluster_vars_scaled)
fviz_cluster(kmeans_result, data = cluster_vars_scaled, geom = "point", ellipse.type = "convex", 
             palette = c("#E41A1C", "#4DAF4A", "#377EB8"), ggtheme = theme_minimal())

# Cluster boxplots
cluster_long <- pivot_longer(
  data_model,
  cols = c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases"),
  names_to = "var",
  values_to = "value",
  values_transform = list(value = as.numeric)
)
ggplot(cluster_long, aes(x = var, y = value, fill = cluster)) +
  geom_boxplot(outlier.shape = 1, position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = c("#E41A1C", "#377EB8", "#4DAF4A")) +
  theme_minimal() +
  labs(title = "Boxplot by Cluster", x = "Variable", y = "Value")


# ------------------- SILHOUETTE ANALYSIS FOR CLUSTER VALIDATION -------------------

library(cluster)  # for silhouette()

# Compute the distance matrix (Euclidean)
dist_matrix <- dist(cluster_vars_scaled)

# Calculate silhouette scores
sil <- silhouette(kmeans_result$cluster, dist_matrix)

# Average silhouette width
avg_sil_width <- mean(sil[, 3])
cat("\n=== Average Silhouette Width ===\n")
cat("Average Silhouette Width:", round(avg_sil_width, 3), "\n")

# Interpret the result
if (avg_sil_width > 0.7) {
  cat("Interpretation: Strong clustering structure\n")
} else if (avg_sil_width > 0.5) {
  cat("Interpretation: Reasonable structure\n")
} else if (avg_sil_width > 0.25) {
  cat("Interpretation: Weak structure, consider re-evaluating number of clusters\n")
} else {
  cat("Interpretation: Poor clustering configuration\n")
}

# Plot silhouette diagram
plot(sil, border = NA, col = c("#E41A1C", "#4DAF4A", "#377EB8"),
     main = "Silhouette Plot for k-means Clustering")


# ---- EVALUATION ----
evaluate <- function(pred, true) {
  pred <- as.numeric(pred)
  true <- as.numeric(true)
  
  RMSE <- sqrt(mean((pred - true)^2, na.rm = TRUE))
  MAE <- mean(abs(pred - true), na.rm = TRUE)
  
  # Create default MAPE in case all true values are zero
  MAPE <- NA
  
  if (any(true != 0)) {
    nonzero <- true != 0
    mape_vals <- abs((pred[nonzero] - true[nonzero]) / true[nonzero])
    # Check for any NaN or Inf
    mape_vals <- mape_vals[is.finite(mape_vals)]
    if (length(mape_vals) > 0) {
      MAPE <- mean(mape_vals) * 100
    }
  }
  
  return(data.frame(RMSE = RMSE, MAE = MAE, MAPE = MAPE))
}


results <- list(
  Linear_Regression = evaluate(lm_pred, test$Purchase.Amount..USD.),
  SVM = evaluate(svm_pred, test$Purchase.Amount..USD.),
  Random_Forest = evaluate(rf_pred, test$Purchase.Amount..USD.),
  Gradient_Boosting = evaluate(gbm_pred, test$Purchase.Amount..USD.),
  XGBoost = evaluate(xgb_pred, test$Purchase.Amount..USD.)
)

print(results)

# ------------------- OPTUNA TUNING FOR XGBOOST (NEW CODE ONLY) -------------------

# Load reticulate
library(reticulate)

# Initialize Python environment (restart R session first if needed)
use_virtualenv("r-reticulate", required = TRUE)

# Ensure Optuna is installed
if (!py_module_available("optuna")) {
  py_install("optuna", pip = TRUE)
}

# Import required Python modules
optuna <- import("optuna")
np <- import("numpy")

# Define the Optuna objective function in R
objective <- function(trial) {
  param <- list(
    eta = trial$suggest_float("eta", 0.01, 0.3),
    max_depth = trial$suggest_int("max_depth", 3, 10),
    subsample = trial$suggest_float("subsample", 0.5, 1.0),
    colsample_bytree = trial$suggest_float("colsample_bytree", 0.5, 1.0),
    lambda = trial$suggest_float("lambda", 0, 5),
    alpha = trial$suggest_float("alpha", 0, 5),
    objective = "reg:squarederror",
    eval_metric = "rmse"
  )
  
  bst <- xgboost::xgb.train(
    params = param,
    data = xgb_train,
    nrounds = 100,
    verbose = 0
  )
  
  preds <- predict(bst, newdata = xgb_test)
  rmse <- sqrt(mean((preds - test$Purchase.Amount..USD.)^2, na.rm = TRUE))
  return(rmse)
}

# Convert R objective to Python-callable function
objective_py <- r_to_py(objective)

# Create and run the Optuna study
study <- optuna$create_study(direction = "minimize")
study$optimize(objective_py, n_trials = 20L)

# Output the best parameters
cat("Best Parameters Found by Optuna:\n")
print(study$best_params)

# Train XGBoost with best parameters
best_param <- study$best_params
best_param$objective <- "reg:squarederror"
best_param$eval_metric <- "rmse"

xgb_tuned <- xgboost::xgb.train(
  params = best_param,
  data = xgb_train,
  nrounds = 100,
  verbose = 0
)

# Predictions and evaluation
xgb_tuned_preds <- predict(xgb_tuned, newdata = xgb_test)
xgb_tuned_results <- evaluate(xgb_tuned_preds, test$Purchase.Amount..USD.)

cat("\n=== Tuned XGBoost ===\n")
print(xgb_tuned_results)

# --- Advanced Clustering with Evaluation (NEW SECTION) ---

# Load required libraries for advanced clustering
library(cluster)
library(dbscan)
library(mclust)

# Function to evaluate clustering
evaluate_clustering <- function(data, clusters) {
  sil <- silhouette(clusters, dist(data))
  avg_sil <- mean(sil[, 3])
  return(list(
    silhouette = avg_sil,
    n_clusters = length(unique(clusters))
  ))
}

# Prepare scaled clustering data
cluster_vars_adv <- data_model[, c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases")]
cluster_vars_adv <- as.data.frame(sapply(cluster_vars_adv, as.numeric))
cluster_vars_adv <- na.omit(cluster_vars_adv)
cluster_vars_scaled_adv <- scale(cluster_vars_adv)

# Compare multiple clustering methods
results <- list()

# K-means
km <- kmeans(cluster_vars_scaled_adv, centers = 3, nstart = 25)
results$kmeans <- evaluate_clustering(cluster_vars_scaled_adv, km$cluster)

# Hierarchical
hc <- hclust(dist(cluster_vars_scaled_adv), method = "ward.D2")
hc_clusters <- cutree(hc, k = 3)
results$hierarchical <- evaluate_clustering(cluster_vars_scaled_adv, hc_clusters)

# PAM (Partitioning Around Medoids)
pam_res <- pam(cluster_vars_scaled_adv, k = 3)
results$pam <- evaluate_clustering(cluster_vars_scaled_adv, pam_res$clustering)

# DBSCAN
db <- dbscan(cluster_vars_scaled_adv, eps = 0.5, minPts = 5)
if (length(unique(db$cluster)) > 1) {
  results$dbscan <- evaluate_clustering(cluster_vars_scaled_adv, db$cluster)
}

# GMM (Gaussian Mixture Models)
gmm <- Mclust(cluster_vars_scaled_adv, G = 3)
results$gmm <- evaluate_clustering(cluster_vars_scaled_adv, gmm$classification)

# Compile results
comparison <- do.call(rbind, lapply(names(results), function(method) {
  data.frame(
    Method = method,
    Silhouette = results[[method]]$silhouette,
    N_Clusters = results[[method]]$n_clusters
  )
}))

print(comparison)

# Optional: Visualize best clustering (e.g., GMM)
fviz_cluster(list(data = cluster_vars_scaled_adv, cluster = gmm$classification),
             geom = "point",
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())


# --- Rule-Based Clustering by Purchase Amount (UPDATED) ---

# Ensure purchase amount is numeric
data_model$Purchase.Amount..USD. <- as.numeric(data_model$Purchase.Amount..USD.)

# Create rule-based clusters
data_model$Purchase_Cluster <- cut(
  data_model$Purchase.Amount..USD.,
  breaks = c(0, 40, 70, 100),
  labels = c("Low", "Mid", "High"),
  include.lowest = TRUE,
  right = FALSE
)

# Check the distribution of rule-based clusters
cat("\n=== Rule-Based Cluster Distribution ===\n")
print(table(data_model$Purchase_Cluster))

# Analyze Promo Code usage by cluster
promo_usage <- data_model %>%
  group_by(Purchase_Cluster, Promo.Code.Used) %>%
  summarise(Count = n()) %>%
  mutate(Percent = round(Count / sum(Count) * 100, 1))

ggplot(promo_usage, aes(x = Purchase_Cluster, y = Percent, fill = Promo.Code.Used)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Promo Code Usage by Purchase Cluster",
    x = "Purchase Cluster",
    y = "Percentage"
  ) +
  scale_fill_manual(values = c("#4DAF4A", "#E41A1C")) +
  theme_minimal()

# Analyze Shipping Type usage by cluster
shipping_usage <- data_model %>%
  group_by(Purchase_Cluster, Shipping.Type) %>%
  summarise(Count = n()) %>%
  mutate(Percent = round(Count / sum(Count) * 100, 1))

ggplot(shipping_usage, aes(x = Purchase_Cluster, y = Percent, fill = Shipping.Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Shipping Type Usage by Purchase Cluster",
    x = "Purchase Cluster",
    y = "Percentage"
  ) +
  theme_minimal()

# --- Statistical Significance Testing by Purchase Cluster ---

cat("\n=== Chi-Squared Test: Promo Code Usage by Purchase Cluster ===\n")
promo_table <- table(data_model$Purchase_Cluster, data_model$Promo.Code.Used)
print(promo_table)
print(chisq.test(promo_table))

cat("\n=== Chi-Squared Test: Shipping Type Usage by Purchase Cluster ===\n")
ship_table <- table(data_model$Purchase_Cluster, data_model$Shipping.Type)
print(ship_table)
print(chisq.test(ship_table))

# Optional: Check normality of Age and Previous Purchases
cat("\n=== Normality Check for Age by Cluster ===\n")
by(data_model$Age, data_model$Purchase_Cluster, shapiro.test)

# If not normal, use Kruskal-Wallis test instead of ANOVA
cat("\n=== Kruskal-Wallis Test: Age by Purchase Cluster ===\n")
print(kruskal.test(Age ~ Purchase_Cluster, data = data_model))

cat("\n=== Kruskal-Wallis Test: Previous Purchases by Purchase Cluster ===\n")
print(kruskal.test(Previous.Purchases ~ Purchase_Cluster, data = data_model))

