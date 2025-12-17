# simple
simple_grid <- expand.grid(
  nrounds = 100,
  max_depth = 4,
  eta = 0.1,
  gamma = 1,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

mod_xgb_simple <- train(
  Exited ~ . + I(Age^2) + Geography:Gender + HasBalance,
  data = train2,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = f1_score,
    verboseIter = TRUE
  ),
  tuneGrid = simple_grid,
  metric = "F1",
  maximize = TRUE,
  scale_pos_weight = pos_weight,
  verbose = FALSE
)