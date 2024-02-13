
# Machine Learning R - Equipo 9.

# Carlos Joaquín Duarte Camacho
# Sergio Iván Medina Martinez
# David Arturo Lozano López
# Ian Balderas Flores
# Abel Isaí Sánchez Nájera
# Victor Enrique Carrizales Méndez


#Importando las librerias utilizadas

pacman::p_load(dplyr,ggplot2, corrplot, caret, tidyverse, rstatix, reshape2, rpart, rpart.plot, ROSE, DescTools, smotefamily, mice) 

#Asignando el dataset a una variable

# Información del Dataset

df_1 <- read.csv("https://raw.githubusercontent.com/disfordavidd/EVC/main/brain_stroke.csv", header = T)

# 1) id: inentificador único
# 2) gender: "Male", "Female" or "Other"
# 3) age: edad del paciente
# 4) hypertension: 0 si el paciente no padece hipertensión, 1 si el paciente tiene hipertensión.
# 5) heart_disease: 0 si el paciente no presenta enfermedades cardiacas, 1 si el paciente presenta enfermedades cardiacas
# 6) ever_married: "No" or "Yes"
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 8) Residence_type: "Rural" or "Urban"
# 9) avg_glucose_level: nivel promedio de glucosa en sangre
# 10) bmi: índice de masa muscular
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
# 12) stroke: 1 if si el paciente tuvo un ictus or 0 si no

#Explorando el df

dim(df_1)
str(df_1)
head(df_1)


#Revisar las columnas del df
colnames(df_1)

# elimina valores outlier
df_clean <- df_1 %>% filter(df_1$gender!='Other')
#df_clean = df_clean %>% filter(df_clean$work_type!='Never_worked')
df_clean = df_clean %>% filter(df_clean$bmi!='N/A')

# convierte chr a factores
df_clean$gender = as.factor(df_clean$gender)
df_clean$ever_married = as.factor(df_clean$ever_married)
df_clean$work_type = as.factor(df_clean$work_type)
df_clean$Residence_type = as.factor(df_clean$Residence_type)
df_clean$smoking_status = as.factor(df_clean$smoking_status)
df_clean$bmi = as.numeric(df_clean$bmi)

# Preprocesamiento de datos. Normaliza variables categóricas
df_num <- data.frame(df_clean)
df_num$ever_married = str_replace_all(df_num$ever_married, c("Yes"="1", "No"="0"))
df_num$ever_married = as.numeric(df_num$ever_married)

df_num$gender = str_replace_all(df_num$gender, c("Male"="1", "Female"="2"))
df_num$gender = as.numeric(df_num$gender)

df_num$work_type = str_replace_all(df_num$work_type, c("Never_worked"="0","children"="1", "Private"="2", "Self-employed"="3", "Govt_job"="4"))
df_num$work_type = as.numeric(df_num$work_type)

df_num$Residence_type = str_replace_all(df_num$Residence_type, c("Rural"="1", "Urban"="2"))
df_num$Residence_type = as.numeric(df_num$Residence_type)

df_num$stroke = as.numeric(as.character(df_num$stroke))

df_num$smoking_status = as.numeric(df_num$smoking_status)

drop_smoke<-c("smoking_status","id")
df_num <- df_num[,!(names(df_num) %in% drop_smoke)]

# Mapa de correlación
res <- cor(df_num)
corrplot(res, type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)


#Mapa de correlaciones con las variables numéricas

#df_corr <- select(df_1, age, avg_glucose_level, bmi)
#df.cor = cor(df_corr)
#corrplot(df.cor)

#Seleccionar columnas de trabajo

df_work <- select(df_1, gender, ever_married, age,work_type,Residence_type,stroke)
head(df_work)
tail(df_work)

#Agrupando por work_type

df_wt <- df_work %>% group_by(work_type) %>%
            summarize(num_infartos = sum(stroke))


#Visualizando el número de infartos reportados para cada tipo de trabajo

g0 <- ggplot(df_wt, aes(x = work_type, y = num_infartos, fill = work_type)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  theme_light() +
  labs(
    title = "Tipo de trabajo y EVC",
    x = "Tipo de Trabajo",
    y = "Número de Infartos"
  )

g0



#Visualizando el % de infartos con respecto al total de infartos 

g1 <- ggplot(df_wt, aes(x = work_type, y = (num_infartos/sum(num_infartos)*100), fill = work_type)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  theme_light() +
  labs(
    title = "Tipo de trabajo y EVC",
    x = "Tipo de Trabajo",
    y = "% del total de Infartos"
  )

g1




#Viendo si existen diferencias entre grupos usando ANOVA

# Perform ANOVA
anova_result <- aov(stroke ~ work_type, data = df_1)

# Summary of ANOVA
summary(anova_result)


#Existen diferencias significativas entre el tipo de trabajo y el número de infartos.

PostHocTest((anova_result), method = "lsd",  conf.level = 0.99)



"TRATAMIENTO DE DATOS PARA MODELO MACHINE LEARNING"

#Factorizando variables categóricas

colnames(df_work)

df_work$work_type <- as.numeric(factor(df_work$work_type))
df_work$gender <- as.numeric(factor(df_work$gender))
df_work$ever_married <- as.numeric(factor(df_work$ever_married))
df_work$Residence_type <- as.numeric(factor(df_work$Residence_type))
df_work$stroke <- factor(df_work$stroke) 

df_work_numeric <- cbind(df_work[, -which(names(df_work) == "stroke")], model.matrix(~stroke - 1, data = df_work))


#Limpiando registros nulos
# df_clean <- na.omit(df_work)
# sum(is.na(df_clean))

#Revisar la distribución de la variable respuesta "stroke" en el dataset

table(df_work$stroke)  #Existe una medida desproporcionada de casos negativos

# Aplicar ROSE para generar muestras sintéticas y equilibrar las clases

df_balanced <- ROSE(stroke ~ ., data = df_clean, N = 600, seed = 123)$data

# Verifica la nueva distribución de clases

table(df_balanced$stroke)

# Dividir datos en conjuntos de entrenamiento y prueba

set.seed(123)  # Establecer semilla para reproducibilidad

datos_entrenamiento <- sample_frac(df_balanced, .7) # 70% para entrenamiento

datos_prueba <- setdiff(df_balanced, datos_entrenamiento)


"MODELO DE ARBOL DE DECISIONES"

#Modelo de árbol 

str(datos_entrenamiento)

str(datos_prueba)

control  <- rpart.control(minsplit=20, minbucket=5, maxdepth=20)

modelo_arbol <- rpart(stroke ~., data = datos_entrenamiento, method = "class", control =  control)
rpart.plot(modelo_arbol)

predicciones <- predict(modelo_arbol, newdata = datos_prueba, type="class")
aux <- factor(datos_prueba$stroke, levels = levels(predicciones))


matriz_confusion <- confusionMatrix(predicciones, aux)
print(matriz_confusion)


#Viendo la importancia de variables

VarImportance <- data.frame(imp = modelo_arbol$variable.importance)

VarImportance

df_importance <- VarImportance %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))

ggplot2::ggplot(df_importance) +
  geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  coord_flip() +
  theme_bw()


"MODELO REGRESION LOGISTICA"

# crea set de entrenamiento-prueba, sin los missing features
# elimina características innecesarias
drop <- c("id","gender", "Residence_type")
df_num = df_num[,!(names(df_num) %in% drop)]

# shuffle index
set.seed(42)
rows<-sample(nrow(df_num))
df_shuffled<-df_num[rows,]

# segmenta datos de entrenamiento y prueba
train <- df_shuffled[1:4000,]
test <- df_shuffled[4001:4908,]

## Modelado
# Modelo de regresión logística.

model <- glm(stroke ~., family=binomial(link='logit'), data=train)
summary(model)
anova(model, test="Chisq")

# accuracy (precisión)
fitted.results <- predict(model,newdata=subset(test,select=c(1,2,3,4,5,6,7)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$stroke)
print(paste('Accuracy',1-misClasificError))
