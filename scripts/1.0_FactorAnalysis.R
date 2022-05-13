##### Set environment #####
rm(list = ls()) # Clear environment
cat("\014") # Clear console
dev.off() # Clear plot window
options(contrasts=c("contr.sum", "contr.poly")) # Set contrast settings to effect coding

# Libraries
library(arrow)
library(lavaan)
library(lavaanPlot)

# Set and Get directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #Set WD to script location

##### Loading data ##### 
imageData <-as.data.frame(read_parquet("../loc_data/df_session_uuid.parquet"))
piscesData <- imageData[imageData$DB == 'PiSCES',]
radboudData <- imageData[imageData$DB == 'Radboud',]
marloesData <- imageData[imageData$DB == 'Marloes',]

library(ltm)
##### Valence #####
piscesDataClean = piscesData[c("ID", "pic_name","valence")]
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
