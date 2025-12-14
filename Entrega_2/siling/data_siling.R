load("dataaaaaaaaaaaaaa.RData")
data_plus_sl <- data_reducida_plus

# --- Balance: log/0-flag/winsorize/bins ---
data_plus_sl$HasBalance <- factor(ifelse(data_plus_sl$Balance > 0, "Yes", "No"),
                                   levels = c("No","Yes"))

data_plus_sl$Balance_log1p <- log1p(data_plus_sl$Balance)

# winsorize: 把极端值截到 1%~99% 分位
q <- quantile(data_plus_sl$Balance, probs = c(0.01, 0.99), na.rm = TRUE)
data_plus_sl$Balance_wins <- pmin(pmax(data_plus_sl$Balance, q[1]), q[2])

# 分箱（让模型更容易学到“高余额更可能流失”这种阈值）
cuts <- quantile(data_plus_sl$Balance, probs = seq(0,1,0.2), na.rm = TRUE)
data_plus_sl$Balance_bin5 <- cut(data_plus_sl$Balance, breaks = unique(cuts),
                                  include.lowest = TRUE, ordered_result = TRUE)

# --- Age non-linear ---
data_plus_sl$Age2 <- data_plus_sl$Age^2
data_plus_sl$Age_bin <- cut(data_plus_sl$Age,
                             breaks = c(-Inf, 30, 40, 50, 60, Inf),
                             labels = c("<=30","31-40","41-50","51-60",">60"),
                             ordered_result = TRUE)

# --- Num products numeric proxy + interactions ---
data_plus_sl$NumProducts_num <- as.numeric(as.character(
  ifelse(data_plus_sl$NumOfProducts_grupo=="1", 1,
         ifelse(data_plus_sl$NumOfProducts_grupo=="2", 2, 3))
))

data_plus_sl$Bal_per_product <- data_plus_sl$Balance / pmax(data_plus_sl$NumProducts_num, 1)

data_plus_sl$Inactive_x_Balance <- ifelse(data_plus_sl$IsActiveMember==0, 1, 0) * data_plus_sl$Balance_log1p
data_plus_sl$Inactive_x_Age <- ifelse(data_plus_sl$IsActiveMember==0, 1, 0) * data_plus_sl$Age

data_plus_sl$Geo_Gender <- interaction(data_plus_sl$Geography, data_plus_sl$Gender, drop = TRUE)
data_plus_sl$Geo_Active <- interaction(data_plus_sl$Geography, data_plus_sl$IsActiveMember, drop = TRUE)
save.image("data_plus_sl.RData")
