library(caret)
library(pROC)
library(ROSE)

set.seed(123)

load("famd_auto.RData") # res_imp, res_red, res_rp, res_t

f1_from_cm <- function(cm){
  p <- as.numeric(cm$byClass["Pos Pred Value"])
  r <- as.numeric(cm$byClass["Sensitivity"])
  if (is.na(p) || is.na(r) || (p+r)==0) return(NA_real_)
  2*p*r/(p+r)
}

myROC <- function(data, lev=NULL, model=NULL){
  twoClassSummary(data, lev = c("Yes","No"))
}

ncp_map <- list(Imputado=19, reducido=8, reducido_plus=10, transformada=28)

# res_obj: res_imp / res_red / res_rp / res_t
run_knn_from_famd <- function(
    res_obj, data_name,
    sampling_vec = c("none","down","up","smote","rose"),
    k_grid_coarse = c(5,7,9),
    k_refine_span = 0
){
  Xtr_full <- as.data.frame(res_obj$Xtr)
  Xte_full <- as.data.frame(res_obj$Xte)
  
  if (data_name %in% names(ncp_map)) {
    ncp_fixed <- ncp_map[[data_name]]
    p <- min(ncp_fixed, ncol(Xtr_full), ncol(Xte_full))
    if (p < ncp_fixed) warning(sprintf("", data_name, ncp_fixed, p))
    Xtr <- Xtr_full[, seq_len(p), drop = FALSE]
    Xte <- Xte_full[, seq_len(p), drop = FALSE]
  } else {
    Xtr <- Xtr_full
    Xte <- Xte_full
  }
  
  ytr <- factor(ifelse(as.character(res_obj$ytr) == "1", "Yes", "No"),
                levels = c("Yes","No"))
  yte <- factor(ifelse(as.character(res_obj$yte) == "1", "Yes", "No"),
                levels = c("Yes","No"))
  colnames(Xtr) <- paste0("F", seq_len(ncol(Xtr)))
  colnames(Xte) <- paste0("F", seq_len(ncol(Xte)))
  
  ## 2) k_best busqueda por 2 etapas
  ctrl_cv_roc <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=myROC)
  fit_coarse <- train(x=Xtr, y=ytr, method="knn",
                      trControl=ctrl_cv_roc, metric="ROC",
                      tuneGrid=data.frame(k=k_grid_coarse))
  k0 <- fit_coarse$bestTune$k
  k_refine <- sort(unique(pmax(1, k0 + (-k_refine_span:k_refine_span))))
  
  fit_base <- train(x=Xtr, y=ytr, method="knn",
                    trControl=ctrl_cv_roc, metric="ROC",
                    tuneGrid=data.frame(k=k_refine))
  k_best <- fit_base$bestTune$k
  
  ## 3) train & eval con k_best
  eval_one <- function(samp){
    ctrl <- if (samp=="none") {
      trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=myROC)
    } else {
      trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=myROC, sampling=samp)
    }
    
    fit <- train(x=Xtr, y=ytr, method="knn",
                 trControl=ctrl, metric="ROC",
                 tuneGrid=data.frame(k=k_best))
    
    # Train
    pred_tr <- predict(fit, newdata=Xtr)
    prob_tr <- predict(fit, newdata=Xtr, type="prob")[,"Yes"]
    roc_tr  <- pROC::roc(response=ytr, predictor=prob_tr, levels=c("No","Yes"), direction="<")
    auc_tr  <- as.numeric(pROC::auc(roc_tr))
    cm_tr   <- confusionMatrix(pred_tr, ytr, positive="Yes")
    
    row_tr <- data.frame(
      DATA=data_name, Split="train", Sampling=samp, Method="KNN",
      ncp_used = ncol(Xtr), k_best = k_best,
      ROC_CV = max(fit$results$ROC, na.rm=TRUE),
      Sens_CV = fit$results$Sens[fit$results$k==k_best],
      Spec_CV = fit$results$Spec[fit$results$k==k_best],
      Accuracy = as.numeric(cm_tr$overall["Accuracy"]),
      Precision = as.numeric(cm_tr$byClass["Pos Pred Value"]),
      Recall = as.numeric(cm_tr$byClass["Sensitivity"]),
      Specificity = as.numeric(cm_tr$byClass["Specificity"]),
      F1 = f1_from_cm(cm_tr),
      AUC = auc_tr,
      stringsAsFactors = FALSE
    )
    
    # TEST
    pred_te <- predict(fit, newdata=Xte)
    prob_te <- predict(fit, newdata=Xte, type="prob")[,"Yes"]
    roc_te  <- pROC::roc(response=yte, predictor=prob_te, levels=c("No","Yes"), direction="<")
    auc_te  <- as.numeric(pROC::auc(roc_te))
    cm_te   <- confusionMatrix(pred_te, yte, positive="Yes")
    
    row_te <- data.frame(
      DATA=data_name, Split="test", Sampling=samp, Method="KNN",
      ncp_used = ncol(Xtr), k_best = k_best,
      ROC_CV = max(fit$results$ROC, na.rm=TRUE),
      Sens_CV = fit$results$Sens[fit$results$k==k_best],
      Spec_CV = fit$results$Spec[fit$results$k==k_best],
      Accuracy = as.numeric(cm_te$overall["Accuracy"]),
      Precision = as.numeric(cm_te$byClass["Pos Pred Value"]),
      Recall = as.numeric(cm_te$byClass["Sensitivity"]),
      Specificity = as.numeric(cm_te$byClass["Specificity"]),
      F1 = f1_from_cm(cm_te),
      AUC = auc_te,
      stringsAsFactors = FALSE
    )
    
    rbind(row_tr, row_te)
  }
  
  sampling_vec <- unique(sampling_vec)
  res <- do.call(rbind, lapply(sampling_vec, eval_one))
  rownames(res) <- NULL
  res
}


metrics_imp <- run_knn_from_famd(res_imp, "Imputado", sampling_vec=c("none","down","up","smote","rose"))
metrics_red <- run_knn_from_famd(res_red, "Reducido", sampling_vec=c("none","down","up","smote","rose"))
metrics_rp  <- run_knn_from_famd(res_rp, "Reducido_plus", sampling_vec=c("none","down","up","smote","rose"))
metrics_t   <- run_knn_from_famd(res_t, "Transformada", sampling_vec=c("none","down","up","smote","rose"))
save.image("knn_con_famd.RData")

