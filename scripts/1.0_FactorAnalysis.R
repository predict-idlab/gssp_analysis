##### Set environment #####
rm(list = ls()) # Clear environment
cat("\014") # Clear console
dev.off() # Clear plot window
options(contrasts=c("contr.sum", "contr.poly")) # Set contrast settings to effect coding

# Libraries
library(arrow)

# Set and Get directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #Set WD to script location

##### Loading data ##### 
imageData <-as.data.frame(read_parquet("../loc_data/df_session_uuid.parquet"))
piscesData <- imageData[imageData$DB == 'PiSCES',]
radboudData <- imageData[imageData$DB == 'Radboud',]
marloesData <- imageData[imageData$DB == 'Marloes',]

# Cronbach's Alpha
library(ltm)
# Arousal
piscesDataClean = piscesData[c("ID", "pic_name","arousal")]
piscesDataClean$pic_name = as.factor(piscesDataClean$pic_name)
piscesDataClean = reshape(piscesDataClean, idvar = "ID", timevar = "pic_name", direction = "wide") # Drops two answers, two people answered 1 trial twice?
cronbach.alpha(piscesDataClean, CI=TRUE, na.rm = TRUE)

# Valence
piscesDataClean = piscesData[c("ID", "pic_name","valence")]
piscesDataClean$pic_name = as.factor(piscesDataClean$pic_name)
piscesDataClean = reshape(piscesDataClean, idvar = "ID", timevar = "pic_name", direction = "wide") # Drops two answers, two people answered 1 trial twice?
cronbach.alpha(piscesDataClean, CI=TRUE, na.rm = TRUE)



# PCA
library(psych)
library(lavaan)
# fa.parallel(piscesData$arousal[,-1], fa="PC", ntrials=100,
#             show.legend=FALSE, main="Scree plot with parallel analysis")

piscesDataClean = piscesData[c("ID", "pic_name","valence","arousal")]
HS.model <- ' pisces  =~ valence + arousal'

fit <- cfa(HS.model, data=piscesDataClean)
summary(fit, fit.measures=TRUE)

# Conclusion: CFA doesn't work with 2 items