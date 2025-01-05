#### Title: Identification of athleticism and sports profiles throughout 
#### machine learning applied to heart rate variability.
#### Author: Tony Estrella
#### October 2024. Reviewed in December 2024

# Libraries ----
library(tidyverse)
library(readxl)
library(tidymodels)
library(compareGroups)
library(themis)
library(DALEXtra)
library(agua)
library(h2o)
library(patchwork)
library(gtsummary)
library(extrafont)


## GGplot2 theme
theme_set(theme_bw())
## Font family
font_import()
loadfonts(device = "win")

# Data --------------------------------------------------------------------

#df<- read_excel("C:/Users/1494617/Dropbox/Doctorat/DA HRV/ML Analysis/data/HRV v10_Total Param UPC_130seg.xls")
df<- read_excel("C:/Users/estre/Dropbox/Doctorat/DA HRV/ML Analysis/data/HRV v10_Total Param UPC_130seg.xls")

# Data preparation. (OUTCOMES)

df$Sport2<-as.factor(df$Sport2) 
#levels(df$Sport2)<- c("Student (non-Athlete)", "Athlete") 

df$Sport<-as.factor(df$Sport)
#levels(df$Sport)<- c("soccer", "hockey", "basket", "student") 

# General Cleaning and Descriptives ------------------------------------------------------------

# Data cleaning for modelling
#Removing innecessary features
df <- df %>% 
  select(-c(Breathing, Team_Club, Gender, Age, Weight, Height, 'Duration(s)', Semafor_3)) 

colnames(df) <- c("Subject", "Sport",
                  "Sport2", "mRR", 
                  "SDNN", "RMSSD",
                  "pNN50", "TI",
                  "HF", "LF",
                  "VLF", "LF/HF",
                  "HFnu", "HFnu_TP",
                  "VLFnu_TP", "LFnu_TP",
                  "SD1", "SD2",
                  "LFnu")


df_desc <- df 

df_desc$Sport2<-as.factor(df_desc$Sport2) 
levels(df_desc$Sport2)<- c("Student (non-Athlete)", "Athlete") 

df_desc$Sport<-as.factor(df_desc$Sport)
levels(df_desc$Sport)<- c("soccer", "hockey", "basket", "student")


df_desc %>% 
  select(-c(Subject, Sport)) %>% 
  tbl_summary(     
    by = Sport2, # outcome
    statistic = list(all_continuous() ~ "{mean} ({sd})",#estadísticas y formato de las columnas continuas
                     all_categorical() ~ "{n} / {N} ({p}%)"), #estadísticas y formato para columnas categóricas
    digits = all_continuous() ~ 2,# redondeo para columnas continuas
    type   = all_categorical() ~ "categorical"# fuerza la visualización de todos los niveles categóricos                                   # cómo deben mostrarse los valores perdidos
  ) %>%
  modify_spanning_header(c("stat_1", "stat_2") ~ "**Model 1**") %>% 
  bold_labels()

df_desc %>% 
  filter(Sport == "soccer") %>% 
  mutate(cas_fut = as.factor(ifelse(Subject == 30, "Soccer Player", "Team"))) %>% 
  #select(-c(Subject, Gender, Breathing, Sport, Sport2, Team_Club, `Duration(s)`, Semafor_3)) %>%
  tbl_summary(     
    by = cas_fut, 
    statistic = list(all_continuous() ~ "{mean} ({sd})",
                     all_categorical() ~ "{n} / {N} ({p}%)"), 
    digits = all_continuous() ~ 2,
    type   = all_categorical() ~ "categorical"                            
  ) %>%
  modify_spanning_header(c("stat_1", "stat_2") ~ "**Model 2**") %>% 
  bold_labels() 



# Algorithms especification ----------------------------------------------------

## Random Forest
rf_spec <- 
  rand_forest(
    mtry = tune(),
    trees = tune(),
    min_n = tune()
  ) %>% 
  set_mode("classification") %>% 
  set_engine("ranger", importance = "impurity")

##XGBoost
xgb_spec <- 
  boost_tree(
    mtry = tune(),
    trees = tune(),
    min_n = tune(),
    learn_rate = tune()
  ) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost", importance = "impurity")

## Support Vector Machine
svm_spec <-
  svm_rbf(
    cost = tune(),
    rbf_sigma = tune()
  ) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# Evaluation metrics
ev_metrics <- metric_set(accuracy, precision, sensitivity, specificity, roc_auc,recall)


# MODEL 1 (Athletes vs. Students) -----------------------------------------

## Data ----
at_df <- df %>% 
  select(-Sport)

### Descriptives with p.values M1
desc_M1 <- compareGroups(Sport2 ~ . -Subject, data= at_df)
createTable(desc_M1, digits = 2)

### Visualization
at_df %>% 
  ggplot(aes(mRR, fill = Sport2))+
  geom_density(alpha = 0.5)+
  labs(x = "mRR",
       fill = "")

at_df %>% 
  ggplot(aes(Sport2, mRR, color = Sport2))+
  geom_jitter(alpha = 0.5)+
  geom_violin(aes(fill= Sport2), alpha = 0.5)+
  labs(x = "Outcome",
       y = "mRR",
       color = "")+
  theme(legend.position = "none")

## Dataset Division ----
set.seed(1996)
at_split <- group_initial_split(at_df, Subject, prop= 0.8, strata = Sport2)
at_training <- training(at_split)
at_test <- testing(at_split)

at_split

### Bootstrap resampling
set.seed(2024)
at_boot <- bootstraps(at_training, strata = Sport2)
at_boot

## Recipe ----
at_rec<- 
  recipe(formula = Sport2 ~  ., 
         data = at_training) %>% 
  step_rm(Subject) %>% 
  step_smote(Sport2, neighbors = 10)

## Workflow ----

### Random Forest
at_wf_rf <- 
  workflow() %>% 
  add_recipe(at_rec) %>% 
  add_model(rf_spec)

### XGBoost
at_wf_xgb <- workflow() %>% 
  add_recipe(at_rec) %>% 
  add_model(xgb_spec)

### SVM
at_wf_svm <- workflow() %>% 
  add_recipe(at_rec) %>% 
  add_model(svm_spec)


## Hyperparameters ----

### Random Forest ----
doParallel::registerDoParallel()

set.seed(1234)
at_hyperparametros_rf <-
  tune_grid(
    object = at_wf_rf,
    resamples = at_boot,
    grid = grid_max_entropy(
      mtry(range  = c(2, 10)),
      min_n(range = c(5, 30)),
      trees(range = c(100, 500)),
      size = 10
    ),
    metrics = ev_metrics,
    control = control_grid(save_pred = TRUE)
  )

#### Visualization Figure S1----
at_hyperparametros_rf %>% 
  collect_metrics() %>% 
  pivot_longer(cols = c('mtry','trees','min_n'),
               names_to = 'hyperparameters',
               values_to = 'Hyperparameters_value') %>% 
  ggplot(aes(Hyperparameters_value, mean, color=mean)) +
  geom_point() +
  facet_grid(.metric~hyperparameters, scales = 'free') +
  labs(y = "Metric mean
       ",
       x = "Hyperparameters value",
       title = "Random Forest hyperparameters search for Model 1",
       color = "Mean")+
  scale_color_viridis_b()+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))



#### BEST CONFIGURATION
at_best_auc_rf <- at_hyperparametros_rf %>% show_best(metric = "roc_auc", n=1)

### XGBoost ----
doParallel::registerDoParallel()

set.seed(1234)

at_hyperparametros_xgb <-
  tune_grid(
    object = at_wf_xgb,
    resamples = at_boot,
    grid = grid_max_entropy(
      mtry(range  = c(2, 10)),
      min_n(range = c(5, 30)),
      trees(range = c(100, 500)),
      learn_rate(),
      size = 10
    ),
    metrics = ev_metrics,
    control = control_grid(save_pred = TRUE)
  )

#### Visualization Figure S3 ----
at_hyperparametros_xgb %>% 
  collect_metrics() %>% 
  pivot_longer(cols = c('mtry','trees','min_n', 'learn_rate'), names_to = 'hyperparameters', values_to = 'Hyperparameters_value') %>% 
  ggplot(aes(Hyperparameters_value, mean, color=mean)) +
  geom_point() +
  facet_grid(.metric~hyperparameters, scales = 'free')+
  labs(y = "Metric mean
       ",
       x = "Hyperparameters value",
       title = "XGBoost hyperparameters search for Model 1",
       color = "Mean")+
  scale_color_viridis_b()+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))



#### BEST CONFIGURATION
at_best_auc_xgb <- at_hyperparametros_xgb %>% show_best(metric = "roc_auc", n=1)

### SVM ----
doParallel::registerDoParallel()

set.seed(1234)

at_hyperparametros_svm <-
  tune_grid(
    object = at_wf_svm,
    resamples = at_boot,
    metrics = ev_metrics,
    control = control_grid(save_pred = TRUE)
  )

#### Visualization Figure S5 ----
at_hyperparametros_svm %>% 
  collect_metrics() %>% 
  pivot_longer(cols = c("cost", "rbf_sigma"), names_to = 'hyperparameters', values_to = 'Hyperparameters_value') %>% 
  ggplot(aes(Hyperparameters_value, mean, color=mean)) +
  geom_jitter() +
  facet_grid(.metric~hyperparameters, scales = 'free') +
  labs(y = "Metric mean
       ",
       x = "Hyperparameters value",
       title = "Support Vector Machine hyperparameters search for Model 1",
       color = "Mean")+
  scale_color_viridis_b()+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))


#### BEST CONFIGURATION
at_best_auc_svm <- at_hyperparametros_svm %>% show_best(metric = "roc_auc", n=1)


## Modelling ----

### Random Forest ----

at_modelo_rf_final <- 
  at_wf_rf %>%
  finalize_workflow(at_best_auc_rf) 

set.seed(201224)
results_at_rf_resample<- fit_resamples(
  at_modelo_rf_final,
  resamples = at_boot,
  metrics = ev_metrics,
  control= control_resamples(save_pred = TRUE))


detailed_metrics <- results_at_rf_resample %>%
  collect_metrics(summarize = FALSE)


metrics_with_ci_rf <- detailed_metrics %>%
  group_by(.metric) %>%
  summarize(
    mean_estimate = mean(.estimate), 
    lower = quantile(.estimate, probs = 0.025),
    upper = quantile(.estimate, probs = 0.975),
    .groups = "drop"
  ) #CI 95%


set.seed(301224) 
at_rf_fit <- at_modelo_rf_final %>% 
  last_fit(at_split, metrics = ev_metrics)

at_rf_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config) %>% 
  mutate(metrica_redondeada = round(.estimate, 2)) %>% 
  knitr::kable()

results_at_rf<- 
  at_rf_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config)

### XGBoost ----

at_modelo_xgb_final <- 
  at_wf_xgb %>%
  finalize_workflow(at_best_auc_xgb)

set.seed(201224)
results_at_xgb_resample<- fit_resamples(
  at_modelo_xgb_final,
  resamples = at_boot,
  metrics = ev_metrics,
  control= control_resamples(save_pred = TRUE))


detailed_metrics_xgb <- results_at_xgb_resample %>%
  collect_metrics(summarize = FALSE)


metrics_with_ci_xgb <- detailed_metrics_xgb %>%
  group_by(.metric) %>%
  summarize(
    mean_estimate = mean(.estimate), 
    lower = quantile(.estimate, probs = 0.025),
    upper = quantile(.estimate, probs = 0.975),
    .groups = "drop"
  ) #CI 95%


set.seed(3012241)
at_xgb_fit <- 
  at_wf_xgb %>%
  finalize_workflow(at_best_auc_xgb) %>% 
  last_fit(at_split, metrics = ev_metrics)

at_xgb_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config) %>% 
  mutate(metrica_redondeada = round(.estimate, 2)) %>% 
  knitr::kable()

results_at_xgb <- 
  at_xgb_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config)

### SVM ----
at_modelo_svm_final <- 
  at_wf_svm %>%
  finalize_workflow(at_best_auc_svm)

set.seed(2012242)
results_at_svm_resample<- fit_resamples(
  at_modelo_svm_final,
  resamples = at_boot,
  metrics = ev_metrics,
  control= control_resamples(save_pred = TRUE))

detailed_metrics_svm <- results_at_svm_resample %>%
  collect_metrics(summarize = FALSE)

metrics_with_ci_svm <- detailed_metrics_svm %>%
  group_by(.metric) %>%
  summarize(
    mean_estimate = mean(.estimate), 
    lower = quantile(.estimate, probs = 0.025),
    upper = quantile(.estimate, probs = 0.975),
    .groups = "drop"
  ) #CI 95%

set.seed(3012242)
at_smv_fit <- 
  at_wf_svm %>%
  finalize_workflow(at_best_auc_svm) %>% 
  last_fit(at_split, metrics = ev_metrics)

at_smv_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config) %>% 
  mutate(metrica_redondeada = round(.estimate, 2)) %>% 
  knitr::kable()

results_at_svm<-
  at_smv_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config)


## Performance visualization
at_results<- data.frame(
  metrica = results_at_rf$.metric,
  RF = results_at_rf$.estimate,
  XGBoost = results_at_xgb$.estimate,
  SVM = results_at_svm$.estimate
)

at_results <-
  at_results %>%
  pivot_longer(cols = c("RF", "XGBoost", "SVM"), 
               names_to = "modelo") %>%
  filter(metrica == "accuracy" |
           metrica == "precision" |
           metrica == "recall" | 
           metrica == "roc_auc")  %>%
  mutate(metrica = factor(metrica), 
         modelo = factor(modelo))

levels(at_results$metrica) <- c("Accuracy", "Precision", "Recall", "ROC AUC")

plot_m1_metrics <- at_results %>% 
  ggplot(aes(x = metrica, y = value, fill = modelo))+
  geom_col(position = "dodge")+ 
  scale_fill_brewer(palette= "Pastel2")+
  scale_y_continuous(limits = c(0,1), expand = c(0,0))+
  labs(y = "", 
       x = "Metric
       
       (a)",
       fill= NULL,
       title= "Model 1
       ")

plot_m1_metrics


## Prediction ----

### Test prediction
at_prediction <- at_smv_fit %>% 
  extract_workflow() %>% 
  predict(at_test, type = "prob")
  

at_prediction <- at_prediction %>% 
  cbind(at_test %>% select(Sport2))

### Curva ROC M1
roc_m1 <- roc_curve(at_prediction, truth = Sport2, .pred_0) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  labs(y = "Sensitivity
       ")+
  scale_y_continuous(expand = c(0.01,0.01))+
  scale_x_continuous(expand = c(0.01,0.01))+
  theme(text = element_text(family = "Palatino", 
                            colour = "black",
                            size = 11,
                            face= "bold"))+
  theme_bw(base_size = 13)

### New prediction 
new_pred <- readxl::read_xlsx("C:/Users/estre/Dropbox/Doctorat/Articulos/Modelos ML HRV/predicciones/M1_prediction.xlsx")

colnames(new_pred) <- c("Subject", 
                        "mRR", "SDNN", "RMSSD",
                        "pNN50", "TI",
                        "HF", "LF", "VLF", "LF/HF",
                        "HFnu", "LFnu", 
                        "HFnu_TP", "VLFnu_TP", "LFnu_TP",
                  "SD1", "SD2")


new_prediction <- at_smv_fit %>% 
  extract_workflow() %>% 
  predict(new_pred, type = "prob")

new_pred <- cbind(new_pred, new_prediction)

new_pred <- new_pred %>% 
  mutate(true = as.factor(c("Non-Athlete", "Non-Athlete", "Athlete", "Athlete")),
         name = as.factor(c("Sedentary", "Patient", "Football Player", "Basketball player"))) %>% 
  pivot_longer(cols = starts_with(".pred"), 
               names_to = "prediction",
               values_to = "Index") %>% 
  select(c(Subject, prediction, Index, true, name)) %>% 
  filter(name != "Patient") #delating the patient subject

new_pred %>% 
  ggplot(aes(Index, name))+
  geom_segment(
    data = new_pred %>% 
      pivot_wider( names_from = prediction,
                   values_from = Index), 
    aes(x = `.pred_Student (non-Athlete)`, xend = .pred_1, 
        y = name, yend = name),
    alpha = 0.7, color = "gray70", size = 1.5)+
  geom_point(aes(color= prediction), size = 5) + 
  scale_x_continuous(limits = c(0, 1), 
                     labels = scales::percent)+
  scale_color_brewer(palette = "Set2", 
                    labels = c("Athlete", "Non-Athlete"))+
  labs(y = NULL, 
       color = "Predicted",
       x = "
       Athleticism Index")+
  theme_bw(base_size = 15)+
  theme(legend.justification = "center",
        legend.background = element_rect(color = 1),
        axis.text.x  = element_text(colour = "black", size = 11, face= "bold"),
        axis.text.y = element_text(colour = "black", size = 11, face= "bold"))

## Interpretation ----
set.seed(0306)

h2o_start()

model_h2o_rf <- rand_forest(
  mtry = 5, 
  trees = 429, 
  min_n = 16) %>% 
  set_engine("h2o", max_runtime_secs = 20) %>% 
  set_mode('classification') %>% 
  fit(Sport2 ~ ., data = at_rec %>% prep () %>% bake(new_data = NULL)) 

int_h2o_rf <- as.h2o(
  at_rec %>% prep() %>% bake(new_data = at_test))

g <- model_h2o_rf %>%  
  extract_fit_engine() %>% 
  h2o::h2o.shap_summary_plot(int_h2o_rf,
                             top_n_features = 10)

shap_m1 <- g + labs(title = "a) Model 1
         ",
         color = "Normalized original value
         ",
         x = "Features
         ",
         y = "SHAP Contribution")+
  theme(plot.title = element_text(hjust = 0))+ 
  guides(color = FALSE)


# MODEL 2 (Football) -----------------------------------------------------------------

## Data ----
foot_df <- df %>% 
  select(-Sport2) %>% 
  filter(Sport == 1) %>% 
  mutate(cas_fut = as.factor(ifelse(Subject == 30, 1, 0)))

### Descriptives with p.values
desc_M2 <- compareGroups(cas_fut ~ . -Subject-Sport, data= foot_df)
createTable(desc_M2, digits = 2)
summary(desc_M2)

### Visualization
foot_df %>%
  ggplot(aes(mRR, fill = cas_fut))+
  geom_density(alpha = 0.5)+
  labs(title= "mRR",
       fill = "")

foot_df %>% 
  ggplot(aes(cas_fut, mRR, color = cas_fut))+
  geom_jitter(alpha= 0.2)+
  geom_violin(aes(fill = cas_fut), alpha = 0.5)+
  labs(x = "",
       y = "mRR")

## Data division ----
set.seed(1425) 

casfut_split <- initial_split(foot_df, prop= 0.8, strata = cas_fut)

casfut_training <- training(casfut_split) 
casfut_test <- testing(casfut_split)  

casfut_split

### Bootstrap resampling
set.seed(2024) 
casfut_boot <- bootstraps(casfut_training) 
casfut_boot

## Recipe ----
casfut_rec <-
  recipe(formula = cas_fut ~  .,
         data = casfut_training) %>% 
  step_rm(Subject, Sport) %>%
  step_smote(cas_fut, neighbors = 10)

## Workflow ----

### RF
casfut_wf_rf <-    
  workflow() %>%
  add_recipe(casfut_rec) %>% 
  add_model(rf_spec)

### XGBoost
casfut_wf_xgb <-    
  workflow() %>%
  add_recipe(casfut_rec) %>% 
  add_model(xgb_spec)

### SVM
casfut_wf_svm <- 
  workflow() %>% 
  add_recipe(casfut_rec) %>% 
  add_model(svm_spec)

## Hyperparameters ----

### RF
set.seed(4564)
doParallel::registerDoParallel()

casfut_hyperparametros_rf <-
  tune_grid(
    object = casfut_wf_rf,
    resamples = casfut_boot,
    grid = grid_max_entropy(
      mtry(range  = c(2, 10)),
      min_n(range = c(5, 30)),
      trees(range = c(100, 500)),
      size = 10),
    metrics = ev_metrics,
    control = control_grid(save_pred = TRUE))

### Visualization
casfut_hyperparametros_rf %>%
  collect_metrics() %>%
  pivot_longer(cols = c('mtry','trees','min_n'),
               names_to = 'hyperparameters',
               values_to = 'Hyperparameters_value') %>%
  ggplot(aes(Hyperparameters_value, mean, color=mean)) +
  geom_point() +
  facet_grid(.metric~hyperparameters, scales = 'free') +
  labs(y = "Metric mean
       ",
       x = "Hyperparameters value",
       title = "Random Forest hyperparameters search for Model 2",
       color = "Mean")+
  scale_color_viridis_b()+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))


#### BEST CONFIGURATION
casfut_best_auc_rf <- casfut_hyperparametros_rf %>%
  show_best(metric = "roc_auc", n = 1)

### XGBoost
doParallel::registerDoParallel()

set.seed(5698)

casfut_hyperparametros_xgb <-
  tune_grid(
    object = casfut_wf_xgb,
    resamples = casfut_boot,
    grid = grid_max_entropy(
      mtry(range  = c(2, 10)),
      min_n(range = c(5, 30)),
      trees(range = c(100, 500)),
      learn_rate(),
      size = 10
    ),
    metrics = ev_metrics,
    control = control_grid(save_pred = TRUE))

### Visualization
casfut_hyperparametros_xgb %>%
  collect_metrics() %>%    
  pivot_longer(cols = c('mtry','trees','min_n', 'learn_rate'), 
               names_to = 'tipo_parametro', 
               values_to = 'valor_parametro') %>%    
  ggplot(aes(valor_parametro, mean, color=mean)) +   
  geom_point() +   
  facet_grid(.metric~tipo_parametro, scales = 'free' ) +   
  labs(y = "Metric mean
       ",
       x = "Hyperparameters value",
       title = "XGBoost hyperparameters search for Model 2",
       color = "Mean")+
  scale_color_viridis_b()+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))


### BEST CONFIGURATION
casfut_mejor_hyper_xgb <- casfut_hyperparametros_xgb %>% show_best(metric = "roc_auc", n = 1)

### SVM
doParallel::registerDoParallel()
set.seed(1234)

casfut_hyperparametros_svm <-
  tune_grid(
    object = casfut_wf_svm,
    resamples = casfut_boot,
    metrics = ev_metrics,
    control = control_grid(save_pred = TRUE)
  )

### Visualization
casfut_hyperparametros_svm %>%
  collect_metrics() %>% 
  pivot_longer(cols = c("cost", "rbf_sigma"), names_to = 'hyperparameters', values_to = 'Hyperparameters_value') %>% 
  ggplot(aes(Hyperparameters_value, mean, color=mean)) +
  geom_jitter() +
  facet_grid(.metric~hyperparameters, scales = 'free') +
  labs(y = "Metric mean
       ",
       x = "Hyperparameters value",
       title = "Support Vector Machine hyperparameters search for Model 2",
       color = "Mean")+
  scale_color_viridis_b()+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))

### BEST CONFIGURATION
casfut_mejor_hyper_svm <- casfut_hyperparametros_svm %>% show_best(metric = "roc_auc", n=1)

## Modelling----

### Random Forest ----
casfut_rf_final <-
  casfut_wf_rf %>%
  finalize_workflow(casfut_best_auc_rf)

set.seed(201224)
results_casfut_rf_resample<- fit_resamples(
  casfut_rf_final,
  resamples = casfut_boot,
  metrics = ev_metrics,
  control= control_resamples(save_pred = TRUE))


detailed_metrics_rf_m2 <- results_casfut_rf_resample %>%
  collect_metrics(summarize = FALSE)


metrics_with_ci_rf_m2 <- detailed_metrics_rf_m2 %>%
  group_by(.metric) %>%
  summarize(
    mean_estimate = mean(.estimate), 
    lower = quantile(.estimate, probs = 0.025),
    upper = quantile(.estimate, probs = 0.975),
    .groups = "drop"
  ) #CI 95%

set.seed(1256)
casfut_rf_fit <-
  casfut_wf_rf %>%
  finalize_workflow(casfut_best_auc_rf) %>%
  last_fit(casfut_split, 
           metrics = ev_metrics)

casfut_rf_fit %>%   
  collect_metrics() %>%
  select(-.estimator,-.config) %>%
  mutate(valor_redondeado = round(.estimate,2)) %>%    
  knitr::kable()


### XGBoost ----
casfut_xgb_final <-
  casfut_wf_xgb %>%
  finalize_workflow(casfut_mejor_hyper_xgb)

set.seed(2012242)
results_casfut_xgb_resample<- fit_resamples(
  casfut_xgb_final,
  resamples = casfut_boot,
  metrics = ev_metrics,
  control= control_resamples(save_pred = TRUE))


detailed_metrics_xgb_m2 <- results_casfut_xgb_resample %>%
  collect_metrics(summarize = FALSE)

metrics_with_ci_xgb_m2 <- detailed_metrics_xgb_m2 %>%
  group_by(.metric) %>%
  summarize(
    mean_estimate = mean(.estimate), 
    lower = quantile(.estimate, probs = 0.025),
    upper = quantile(.estimate, probs = 0.975),
    .groups = "drop"
  ) #CI 95%

set.seed(1256)
casfut_xgb_fit <-
  casfut_wf_xgb %>%
  finalize_workflow(casfut_mejor_hyper_xgb) %>%
  last_fit(casfut_split, 
           metrics = ev_metrics)

casfut_xgb_fit %>%   
  collect_metrics() %>%
  select(-.estimator,-.config) %>%
  mutate(valor_redondeado = round(.estimate,2)) %>%    
  knitr::kable()

### SVM ----
casfut_svm_final <-
  casfut_wf_svm %>%
  finalize_workflow(casfut_mejor_hyper_svm)

set.seed(2012243)
results_casfut_svm_resample<- fit_resamples(
  casfut_svm_final,
  resamples = casfut_boot,
  metrics = ev_metrics,
  control= control_resamples(save_pred = TRUE))


detailed_metrics_svm_m2 <- results_casfut_svm_resample %>%
  collect_metrics(summarize = FALSE)

metrics_with_ci_svm_m2 <- detailed_metrics_svm_m2 %>%
  group_by(.metric) %>%
  summarize(
    mean_estimate = mean(.estimate), 
    lower = quantile(.estimate, probs = 0.025),
    upper = quantile(.estimate, probs = 0.975),
    .groups = "drop"
  ) #CI 95%

set.seed(1256)
casfut_svm_fit <-
  casfut_wf_svm %>%
  finalize_workflow(casfut_mejor_hyper_svm) %>%
  last_fit(casfut_split, 
           metrics = ev_metrics)

casfut_svm_fit %>%   
  collect_metrics() %>%
  select(-.estimator,-.config) %>%
  mutate(valor_redondeado = round(.estimate,2)) %>%    
  knitr::kable()

### Visualization
results_modelo2_rf<-
  casfut_rf_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config) 

results_modelo2_xgb<-
  casfut_xgb_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config) 

results_modelo2_svm<-
  casfut_svm_fit %>% 
  collect_metrics() %>% 
  select(-.estimator, -.config) 


casm2_results<- data.frame(
  metrica = results_modelo2_rf$.metric,
  RF = results_modelo2_rf$.estimate,
  XGBoost = results_modelo2_xgb$.estimate,
  SVM = results_modelo2_svm$.estimate
)

casm2_results <-
  casm2_results %>%
  pivot_longer(cols = c("RF", "XGBoost", "SVM"), 
               names_to = "modelo") %>%
  filter(metrica == "accuracy" |
           metrica == "precision" |
           metrica == "recall" | 
           metrica == "roc_auc")  %>%
  mutate(metrica = factor(metrica), 
         modelo = factor(modelo))

casm2_results

levels(casm2_results$metrica) <- c("Accuracy", "Precision", "Recall", "ROC AUC")


plot_m2_metrics <- casm2_results %>% 
  ggplot(aes(x = metrica, y = value, fill = modelo))+
  geom_col(position = "dodge")+ 
  scale_fill_brewer(palette= "Pastel2")+
  scale_y_continuous(limits = c(0,1), expand = c(0,0))+
  labs(y = "", 
       x = "Metric
       
       (b)",
       fill= NULL,
       title= "Model 2
       ")
## Prediction ----

### Test prediction
cas_prediction <- casfut_rf_fit %>% 
  extract_workflow() %>% 
  predict(casfut_test, type = "prob")


cas_prediction<- cas_prediction %>% 
  mutate(true_value = casfut_test$cas_fut)
cas_prediction

### Curva ROC M2
roc_m2 <- roc_curve(cas_prediction, truth = true_value, .pred_0) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  labs(y = "
       ")+
  scale_y_continuous(expand = c(0.01,0.01))+
  scale_x_continuous(expand = c(0.01,0.01))+
  theme(text = element_text(family = "Palatino", 
                            colour = "black",
                            size = 11,
                            face= "bold"))+
  theme_bw(base_size = 13)

### Explanation of misspredictions

test_pred <- cbind(casfut_test, cas_prediction) #adding probability and true value


## Interpretation ----
set.seed(1104)

h2o_start()

modelo3_h2o_rf <- rand_forest(
  mtry = 2,
  trees = 214,
  min_n = 10
) %>% 
  set_engine("h2o", max_runtime_secs = 20) %>% 
  set_mode('classification') %>% 
  fit(cas_fut ~ ., data = casfut_rec %>% prep () %>% bake(new_data = NULL)) 



int3_h2o_rf <- h2o::as.h2o(
  casfut_rec %>% prep() %>% bake(new_data = NULL))


g2 <- modelo3_h2o_rf %>%  
  extract_fit_engine() %>% 
  h2o::h2o.shap_summary_plot(int3_h2o_rf,
                             top_n_features = 10) #top_n_features = 10

shap_m2 <- g2 + labs(title = "b) Model 2",
          color = "Normalized original value
         ",
          x = "
         ", 
          y = "SHAP contribution")+
  theme(plot.title = element_text(hjust = 0))


# Figure 2 ----
plot_m1_metrics + plot_m2_metrics

# Figure 3----
### These figures were updated after the first peer-review

fig3a <- at_prediction %>%
  mutate(True = as.factor(at_test$Sport2)) %>%
  arrange(desc(.pred_1)) %>% 
  mutate(n = as.factor(seq(1, length(.pred_1)))) %>% 
  pivot_longer(
    cols = starts_with(".pred"),
    names_to = "prediction",
    values_to = "Index") %>%
  filter(True == 1) %>% 
  ggplot(aes(x = n, y = Index, fill = as.factor(prediction)))+
  geom_col(position = position_stack(reverse = TRUE), 
           alpha = .5,
           width = 1)+
  geom_hline(yintercept = 0.5,
             linetype = "dashed")+
  scale_fill_brewer(palette = "Set2")+
  theme(axis.title.y=element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y= element_blank()) + 
  labs(title = "True Athlete (n = 165)") +
  scale_y_continuous(expand = c(0.01,0.01)) +
  guides(fill = FALSE)+
  coord_flip()


fig3b <- at_prediction %>%
  mutate(True = as.factor(at_test$Sport2)) %>%
  arrange(.pred_1) %>% 
  mutate(n = as.factor(seq(1, length(.pred_1)))) %>% 
  pivot_longer(
    cols = starts_with(".pred"),
    names_to = "prediction",
    values_to = "Index") %>%
  filter(True != "Athlete") %>% 
  ggplot(aes(x = n, y = Index, fill = as.factor(prediction)))+
  geom_col(position = "stack", 
           alpha = .5,
           width = 1)+
  geom_hline(yintercept = 0.5,
             linetype = "dashed")+
  scale_fill_brewer(palette = "Set2", 
                    labels = c("Athlete", "Student"))+
  theme(axis.title.y=element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y= element_blank(), 
        legend.title = element_text(face = "bold"),
        legend.position = "bottom",
        legend.background = element_rect(color = "black")) + 
  labs(title = "True Student (n = 99)",
       fill = "Prediction") +
  scale_y_continuous(expand = c(0.01,0.01)) +
  coord_flip()


fig3a / fig3b


## After first peer-review

## Figure 3 reviewed ----
fig3_updated <- at_prediction %>% 
  group_by(Sport2) %>% 
  summarise(mean_pred1= mean(.pred_1),
            sd_pred1 = sd(.pred_1),
            n = length(.pred_1),
            se = sd_pred1/(sqrt(length(.pred_1)))) %>% 
  ggplot(aes(x = Sport2, mean_pred1*100))+
  geom_col(aes(fill= Sport2),
           color = "black")+
  geom_text(aes(label = paste(round(mean_pred1*100, 2), "%" )), vjust= 5, size = 8, family = "Palatino Linotype")+
  geom_errorbar(aes(ymin = mean_pred1*100 - se*100, ymax = mean_pred1*100 + se*100),
                linewidth = 0.8,
                width= .3, 
                alpha = 0.8)+
  labs(x = "",
       y = "Athleticism index
       ",
       title = "")+
  scale_y_continuous(limits = c(0,100),
                     label = c("0%", "25%", "50%", "75%", "100%"),
                     expand = c(0,0))+
  scale_x_discrete(limits = rev(levels(at_prediction$Sport2)),
                   labels = c("True athletes", "True students"))+
  scale_fill_brewer(palette="Pastel1")+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(legend.position = "none",
        axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))

  

# Figure 4 ----
## This figure was updated after first peer-review
new_pred %>% 
  ggplot(aes(Index, name))+
  geom_segment(
    data = new_pred %>% 
      pivot_wider( names_from = prediction,
                   values_from = Index), 
    aes(x = .pred_0, xend = .pred_1, 
        y = name, yend = name),
    alpha = 0.7, color = "gray70", size = 1.5)+
  geom_point(aes(color= prediction), size = 5) + 
  scale_x_continuous(limits = c(0, 1), 
                     labels = scales::percent)+
  scale_color_brewer(palette = "Set2", 
                     labels = c("Athlete", "Non-Athlete"))+
  labs(y = NULL, 
       color = "Predicted",
       x = "
       Athleticism Index")+
  theme_bw(base_size = 15)+
  theme(legend.justification = "center",
        legend.background = element_rect(color = 1),
        axis.text.x  = element_text(colour = "black", size = 11, face= "bold"),
        axis.text.y = element_text(colour = "black", size = 11, face= "bold"))

## After first peer-review
## Figure 4 reviewed ----

df_fig4 <- data.frame( tipo = factor(rep(c("Sedentary Person", "Soccer Player", "Basketball Player"), 2)),
                       categoria = factor(rep(c("Athlete", 
                                                "Non-Athlete"), 3)),
                       value = c(12.8, 17.6, 92.2, 87.2, 82.4, 7.8))
df_fig4 %>%
  filter(categoria == "Athlete") %>% 
  ggplot(aes(reorder(tipo, value), value))+ # reordenar eje X
  geom_col(fill = "#acdacc", color = "black")+
  geom_text(aes(label = paste(value, "%" )), vjust= -0.3, size = 5, family = "Palatino Linotype")+
  scale_y_continuous(limits = c(0,100), label = c("0%", "25%", "50%", "75%", "100%"),
                     expand = c(0,0))+
  labs(y = "Athleticism index
       ", 
       x = "",
       title = "")+
  theme_bw(base_size = 15,
           base_family = "Palatino Linotype")+
  theme(axis.text.x  = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"),
        axis.text.y = element_text(family = "Palatino Linotype", colour = "black", size = 10, face= "bold"))

# Figure 5----

shap_m1 + shap_m2




