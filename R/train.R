library(h2o)
h2o.init(nthreads=-1, max_mem_size='5G')
h2o.removeAll()

setwd('~/Projects/pubNative/R')

train <- read.csv('../db/train.csv', stringsAsFactors=F)
valid <- read.csv('../db/valid.csv', stringsAsFactors=F)

# Check summary of datasets
summary(train)
summary(valid)

logical_columns <- c('v1', 'v9', 'v10', 'v12', 'v19', 'classLabel')

train[logical_columns][is.na(train[logical_columns])] <- F
valid[logical_columns][is.na(valid[logical_columns])] <- F

# Correct boolean datatype columns
for(col in logical_columns) {
	train[train[[col]] == 'False', c(col)] <- F
	train[train[[col]] == 'True', c(col)] <- T
	train[train[[col]] == 0, c(col)] <- F
	train[train[[col]] == 1, c(col)] <- T
	train[[col]] <- as.logical(train[[col]])
	
	valid[valid[[col]] == 'False', c(col)] <- F
	valid[valid[[col]] == 'True', c(col)] <- T
	valid[valid[[col]] == 0, c(col)] <- F
	valid[valid[[col]] == 1, c(col)] <- T
	valid[[col]] <- as.logical(valid[[col]])
}

categorical_columns <- c('v4', 'v5', 'v6', 'v7', 'v13')
for(col in categorical_columns) {
	train[[col]] <- as.factor(train[[col]])
	valid[[col]] <- as.factor(valid[[col]])
}

not_fs <- c('v19')

train_h2o <- as.h2o(train)
valid_h2o <- as.h2o(valid)
	
response <- 'classLabel'
predictors <- setdiff(names(train), response)
predictors <- setdiff(predictors, not_fs)
	
rf1 <- h2o.randomForest(
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  x=predictors,
  y=response,
  model_id = 'rf1',
  ntrees = 50,
  max_depth = 5,
  stopping_rounds = 2,
  stopping_tolerance = 1e-2,
  score_each_iteration = T,
  seed=10)
	  
accuracy <- h2o.auc(rf1, valid=T) * 100
accuracy = h2o.logloss(rf1, valid=T)
accuracy <- round(accuracy, 2)
print(paste('LogLoss:', accuracy))
	  
h2o.saveModel(object=rf1, path=getwd(), force=TRUE)

varimp <- as.data.frame(h2o.varimp(rf1))

# -------- #
# Try Gradient Boosting machine #

# Hyperparameter tuning

# GBM hyperparamters
gbm_params1 <- list(learn_rate = c(0.01, 0.1, 0.5),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.5, 0.8, 1.0),
                    col_sample_rate = c(0.1, 0.2, 0.5))

# Train and validate a cartesian grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = predictors, y = response,
                      grid_id = "gbm_grid1",
                      training_frame = train_h2o,
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by validation LogLoss
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1",
                             sort_by = "logloss",
                             decreasing = FALSE)
                          
print(gbm_gridperf1)   
                             
                
# Grab the top GBM model, chosen by validation LogLoss
best_gbm1 <- h2o.getModel(gbm_gridperf1@model_ids[[1]]) 

summary(best_gbm1)            
                             

# Variable importances
varimp <- as.data.frame(h2o.varimp(best_gbm1))
print(varimp)

# CHosen model

gbm1 = h2o.gbm(x = predictors,
               y = response,
			   training_frame = train_h2o,
			   validation_frame = valid_h2o,
               ntrees = 100,
               col_sample_rate = 0.2,
               max_depth = 5,
               min_rows = 5,
               sample_rate = 1.0,
               learn_rate = 0.1,
               keep_cross_validation_predictions = TRUE,
               seed = 30)

# accuracy <- h2o.auc(gbm1, valid=T) * 100
accuracy = h2o.logloss(gbm1, valid=T)
accuracy <- round(accuracy, 2)
print(paste('LogLoss:', accuracy))




# -------- # 

gbm_params1 <- list(learn_rate = c(0.1),
                    max_depth = c(5),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2),
                    min_rows = c(5, 7, 10))

# Train and validate a cartesian grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = predictors, y = response,
                      grid_id = "gbm_grid1",
                      training_frame = train_h2o,
                      validation_frame = valid_h2o,
                      ntrees = 100,
                      seed = 30,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by validation LogLoss
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1",
                             sort_by = "logloss",
                             decreasing = FALSE)
                          
print(gbm_gridperf1)   
                             
# ------------------- #

# Next chosen model

gbm1 = h2o.gbm(x = predictors,
               y = response,
			   training_frame = train_h2o,
			   validation_frame = valid_h2o,
               ntrees = 100,
               col_sample_rate = 0.2,
               max_depth = 5,
               sample_rate = 0.8,
               learn_rate = 0.1,
               keep_cross_validation_predictions = TRUE,
               seed = 3)


logloss = h2o.logloss(gbm1, valid=T)
logloss <- round(logloss, 2)
print(paste('LogLoss:', logloss))

auc <- h2o.auc(gbm1, valid=T) * 100
auc <- round(auc, 2)
print(paste('AUC:', auc, '%'))

