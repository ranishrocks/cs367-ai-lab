# Load required libraries
library(bnlearn)
library(caret)
library(e1071)

# Read the data
data <- read.table("E:/CS307/2020_bn_nb_data.txt", header = TRUE, sep = "\t")

# Convert character columns to factors
data[] <- lapply(data, function(x) if(is.character(x)) as.factor(x) else x)

# 1. Learn dependencies between courses
bn_structure <- hc(data[, 1:8])  # Using hill-climbing algorithm
plot(bn_structure)

# 2. Learn CPTs for each course node
bn_fit <- bn.fit(bn_structure, data[, 1:8])

# 3. Predict grade in PH100 given specific conditions
# Check parents of PH100 in the Bayesian network
parents_PH100 <- parents(bn_structure, "PH100")
print(paste("Parents of PH100:", paste(parents_PH100, collapse = ", ")))

# Prepare the evidence data frame with only the relevant parent variables
evidence_data <- data.frame(matrix(ncol = length(parents_PH100), nrow = 1))
colnames(evidence_data) <- parents_PH100

# Set the evidence values
for (parent in parents_PH100) {
  if (parent == "EC100") evidence_data[1, parent] <- "DD"
  else if (parent == "IT101") evidence_data[1, parent] <- "CC"
  else if (parent == "MA101") evidence_data[1, parent] <- "CD"
  else evidence_data[1, parent] <- NA  # For any other parent variables
}

# Convert character columns to factors with the same levels as in the original data
evidence_data[] <- lapply(names(evidence_data), function(col) {
  factor(evidence_data[[col]], levels = levels(data[[col]]))
})

# Predict the grade in PH100 based on the evidence
predicted_grade <- predict(bn_fit, node = "PH100", data = evidence_data)
print(paste("Predicted grade in PH100:", predicted_grade))

# Calculate the probability distribution for PH100
grade_probabilities <- predict(bn_fit, node = "PH100", data = evidence_data, prob = TRUE)
print("Probability distribution for PH100:")
print(grade_probabilities)

# 4. Naive Bayes classifier (assuming independence)
# Naive Bayes classifier (assuming independence)
naive_bayes_results <- replicate(20, {
  # Split data into training (70%) and testing (30%) sets
  train_indices <- createDataPartition(data$QP, p = 0.7, list = FALSE)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Train Naive Bayes classifier
  # Ensure to use the correct reference for 'QP'
  nb_model <- naiveBayes(QP ~ ., data = train_data)
  
  # Make predictions on test data
  predictions <- predict(nb_model, test_data)
  
  # Calculate accuracy
  accuracy <- mean(predictions == test_data$QP)
  return(accuracy)
})

print(paste("Mean accuracy (Naive Bayes):", mean(naive_bayes_results)))

# 5. Bayesian Network classifier (considering dependencies)
bn_results <- replicate(20, {
  # Split data into training (70%) and testing (30%) sets
  train_indices <- createDataPartition(data$QP, p = 0.7, list = FALSE)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Learn Bayesian Network structure and parameters
  bn_structure <- hc(train_data)
  bn_fit <- bn.fit(bn_structure, train_data)
  
  # Make predictions on test data
  predictions <- predict(bn_fit, node = "QP", test_data[, -9])
  
  # Calculate accuracy
  accuracy <- mean(predictions == test_data$QP)
  return(accuracy)
})

print(paste("Mean accuracy (Bayesian Network):", mean(bn_results)))

