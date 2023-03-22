##### Set environment #####
rm(list = ls()) # Clear environment
cat("\014") # Clear console
dev.off() # Clear plot window
options(contrasts=c("contr.sum", "contr.poly")) # Set contrast settings to effect coding

# Libraries
library(arrow)
library(lavaan)
library(lavaanPlot)
library(ltm)

# Set and Get directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #Set WD to script location

##### Loading data #####
imageData <-as.data.frame(read_parquet("../loc_data/df_session_tot_cleaned.parquet"))

piscesData <- imageData[imageData$DB == 'PiSCES',]
radboudData <- imageData[imageData$DB == 'Radboud',]
marloesData <- imageData[imageData$DB == 'marloes',]

##### Valence #####
piscesDataClean = piscesData[c("ID", "pic_name","valence")]
piscesDataClean$pic_name = as.factor(piscesDataClean$pic_name)
piscesDataClean = reshape(piscesDataClean, idvar = "ID", timevar = "pic_name", direction = "wide") # Drops two answers, two people answered 1 trial twice?

# Cronbach's Alpha
cronbach.alpha(piscesDataClean, CI=TRUE, na.rm = TRUE)
library(psych)
piscesDataCronbachs = piscesDataClean[ ,2:16]
alphavar = alpha(piscesDataCronbachs, check.keys = TRUE)
summary(alphavar)
# CFA
names(piscesDataClean)[2:16] = c("Picture_105", "Picture_82",  "Picture_118", "Picture_65", "Picture_88", "Picture_87", "Picture_59", "Picture_93", "Picture_56", "Picture_81",
                                 "Picture_110", "Picture_96",  "Picture_132", "Picture_80",  "Picture_98" )

HS.model <- 'pisces =~ Picture_105 + Picture_82 + Picture_118 + Picture_65 + Picture_88 + Picture_87 + Picture_59 + Picture_93 + Picture_56 + Picture_81 + Picture_110 + Picture_96 + Picture_132 + Picture_80 + Picture_98'

fit <- cfa(HS.model, data=piscesDataClean, std.lv=TRUE)
summary(fit, fit.measures=TRUE, standardized=TRUE)

# Visualize CFA
labels <- list(Picture_105 = "P_105", Picture_82 = "P_82", Picture_118 = "P_118", Picture_65 = "P_65", Picture_88 = "P_88", Picture_87 = "P_87", Picture_59 = "P_59", Picture_93 = "P_93", Picture_56 = "P_56", Picture_81 = "P_81", Picture_110 = "P_110", Picture_96 = "P_96", Picture_132 = "P32", Picture_80 = "P_80", Picture_98 = "P_98")

# Source: https://cran.r-project.org/web/packages/lavaanPlot/vignettes/Intro_to_lavaanPlot.html
# Basic
lavaanPlot(model = fit, labels = labels, node_options = list(shape = "box", fontname = "Helvetica"), edge_options = list(color = "grey"), coefs = TRUE, sig = .05)
# Standardized and with significance levels
lavaanPlot(model = fit, labels = labels, node_options = list(shape = "box", fontname = "Helvetica"), edge_options = list(color = "grey"), coefs = TRUE, stand = TRUE, stars = "latent")

##### Arousal #####
piscesDataClean = piscesData[c("ID", "pic_name","arousal")]
piscesDataClean$pic_name = as.factor(piscesDataClean$pic_name)
piscesDataClean = reshape(piscesDataClean, idvar = "ID", timevar = "pic_name", direction = "wide") # Drops two answers, two people answered 1 trial twice?

# Cronbach's Alpha
cronbach.alpha(piscesDataClean, CI=TRUE, na.rm = TRUE)

# CFA
names(piscesDataClean)[2:16] = c("Picture_105", "Picture_82",  "Picture_118", "Picture_65", "Picture_88", "Picture_87", "Picture_59", "Picture_93", "Picture_56", "Picture_81",
                                 "Picture_110", "Picture_96",  "Picture_132", "Picture_80",  "Picture_98" )

HS.model <- 'pisces =~ Picture_105 + Picture_82 + Picture_118 + Picture_65 + Picture_88 + Picture_87 + Picture_59 + Picture_93 + Picture_56 + Picture_81 + Picture_110 + Picture_96 + Picture_132 + Picture_80 + Picture_98'

fit <- cfa(HS.model, data=piscesDataClean, std.lv=TRUE)
summary(fit, fit.measures=TRUE, standardized=TRUE)

# Visualize CFA
labels <- list(Picture_105 = "P_105", Picture_82 = "P_82", Picture_118 = "P_118", Picture_65 = "P_65", Picture_88 = "P_88", Picture_87 = "P_87", Picture_59 = "P_59", Picture_93 = "P_93", Picture_56 = "P_56", Picture_81 = "P_81", Picture_110 = "P_110", Picture_96 = "P_96", Picture_132 = "P32", Picture_80 = "P_80", Picture_98 = "P_98")

# Source: https://cran.r-project.org/web/packages/lavaanPlot/vignettes/Intro_to_lavaanPlot.html
# Basic
lavaanPlot(model = fit, labels = labels, node_options = list(shape = "box", fontname = "Helvetica"), edge_options = list(color = "grey"), coefs = TRUE, sig = .05)
# Standardized and with significance levels
lavaanPlot(model = fit, labels = labels, node_options = list(shape = "box", fontname = "Helvetica"), edge_options = list(color = "grey"), coefs = TRUE, stand = TRUE, stars = "latent")


###### Other stuff
library(car)
piscesDataClean = piscesData[c("ID", "pic_name","valence")] # Get long data again

min(piscesDataClean$valence, na.rm=TRUE)
max(piscesDataClean$valence, na.rm=TRUE)
densityPlot(piscesDataClean$valence, na.rm=TRUE)

##### Correlation #####
piscesDataClean = piscesData[c("ID", "sex", "age", "pic_name","arousal","valence")]
piscesDataClean$pic_name = as.factor(piscesDataClean$pic_name)
piscesDataClean$ID = as.factor(piscesDataClean$ID)
piscesDataClean$sex = as.factor(piscesDataClean$sex)
# piscesDataClean = reshape(piscesDataClean, idvar = "ID", timevar = "pic_name", direction = "wide") # Drops two answers, two people answered 1 trial twice?
cor(piscesDataClean$arousal, piscesDataClean$valence,  method = "pearson", use = "complete.obs")
library("ggpubr")
ggscatter(piscesDataClean, x = "arousal", y = "valence",
          add = "reg.line", conf.int = TRUE,
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "arousal", ylab = "valence")

# GLM
options(contrasts = c("contr.sum","contr.poly")) #use this for the p value of the t test
library(lme4)
library(lmerTest)
library(emmeans)
library(effects)

# Arousal
formula <- 'arousal ~ pic_name + (1|ID)' # Declare formula

d0.1 <- lmer(formula,data=piscesDataClean)
# d0.2 <- glmer(formula,data=piscesDataClean, family = Gamma(link = "identity"),glmerControl(optimizer= "bobyqa", optCtrl = list(maxfun = 100000)),nAGQ = 1)
# d0.3 <- glmer(formula,data=piscesDataClean, family = inverse.gaussian(link = "identity"),glmerControl(optimizer= "bobyqa", optCtrl = list(maxfun = 100000)),nAGQ = 1)

# Model Selection
modelNames = c(d0.1)
tabel <- cbind(AIC(d0.1))
chosenModel = modelNames[which(tabel == min(tabel))] # Get model with lowest AIC

Anova(chosenModel[[1]], type = 'III')

emmeans0.1 <- emmeans(chosenModel[[1]], pairwise ~ pic_name, adjust ="fdr", type = "response") #we don't adjust because we do this later
emm0.1 <- summary(emmeans0.1)$emmeans
emmeans0.1$contrasts

plot(effect("pic_name", chosenModel[[1]]), )
plot(effect("pic_name", chosenModel[[1]]), las = 2)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(effect("pic_name", chosenModel[[1]]))

library(ggstatsplot)
library(Polychrome)

# for reproducibility
set.seed(123)

color = grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
n = 15 
colPalette = sample(color, n)
# P15 = createPalette(n,  c("#ff0000", "#00ff00", "#0000ff"))
pie(rep(1,n), col = colPalette)

# plot
p = ggstatsplot::ggbetweenstats(
  data = piscesDataClean,
  x = pic_name,
  y = arousal,
  notch = TRUE,                                   # show notched box plot
  mean.plotting = TRUE,                           # whether mean for each group id to be displayed 
  type = "parametric",                            # which type of test is to be run
  xlab = "Image number",                          # label for the x-axis variable
  ylab = "Arousal",                               # label for the y-axis variable
  title = "Arousal values",                       # title text for the plot
  ggtheme = ggplot2::theme_grey(),                # choosing a different theme
  # palette = "Set3" + c("#00AFBB", "#E7B800", "#FC4E07"),                              # choosing a different color palette
  messages = FALSE
) 
p = p + scale_color_manual(values = colPalette)
p

# Valence
formula <- 'valence ~ pic_name + (1|ID)' # Declare formula

d0.1 <- lmer(formula,data=piscesDataClean)
d0.2 <- glmer(formula,data=piscesDataClean, family = Gamma(link = "identity"),glmerControl(optimizer= "bobyqa", optCtrl = list(maxfun = 100000)),nAGQ = 1)
d0.3 <- glmer(formula,data=piscesDataClean, family = inverse.gaussian(link = "identity"),glmerControl(optimizer= "bobyqa", optCtrl = list(maxfun = 100000)),nAGQ = 1)

# Model Selection
modelNames = c(d0.1,d0.2,d0.3)
tabel <- cbind(AIC(d0.1), AIC(d0.2), AIC(d0.3))
chosenModel = modelNames[which(tabel == min(tabel))] # Get model with lowest AIC

Anova(chosenModel[[1]], type = 'III')

emmeans0.1 <- emmeans(chosenModel[[1]], pairwise ~ pic_name, adjust ="fdr", type = "response") #we don't adjust because we do this later
emm0.1 <- summary(emmeans0.1)$emmeans
emmeans0.1$contrasts

plot(effect("pic_name", chosenModel[[1]]))

# plot
p = ggstatsplot::ggbetweenstats(
  data = piscesDataClean,
  x = pic_name,
  y = valence,
  notch = TRUE,                                   # show notched box plot
  mean.plotting = TRUE,                           # whether mean for each group id to be displayed 
  type = "parametric",                            # which type of test is to be run
  xlab = "Image number",                          # label for the x-axis variable
  ylab = "Valence",                               # label for the y-axis variable
  title = "Valence values",                       # title text for the plot
  ggtheme = ggplot2::theme_grey(),                # choosing a different theme
  # palette = "Set3" + c("#00AFBB", "#E7B800", "#FC4E07"),                              # choosing a different color palette
  messages = FALSE
) 
p = p + scale_color_manual(values = colPalette)
p

# Density plots
library(car)
densityPlot(piscesDataClean$valence, na.rm=TRUE)
densityPlot(piscesDataClean$arousal, na.rm=TRUE)


# 
##### Valence #####
radboudDataClean = radboudData[c("ID", "pic_name","valence")]
radboudDataClean$pic_name = as.factor(radboudDataClean$pic_name)
radboudDataClean = reshape(radboudDataClean, idvar = "ID", timevar = "pic_name", direction = "wide")
radboudDataCronbachs = radboudDataClean[ ,2:16]
