##############################
#                            #
#        1.1_Stats.R         #
#       GSST-Paradigm        #
#        Statistics          #
#     Jitter & Shimmer       #
#                            #
#############################
# 
# Author: Mitchel Kappen 
# 22-3-2023

library(arrow) # Parquets
library(lme4)
library(car)
library(emmeans)
library(ggplot2)
library(dplyr)
library(effects)
library(ggpubr)
library(psych) # Cohen.d

##### Set environment #####
rm(list = ls()) # Clear environment
cat("\014") # Clear console # # Or ctrl + l in VSCode
dev.off() # Clear plot window

# Set and Get directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #Set WD to script location

options(contrasts = c("contr.sum","contr.poly")) #use this for the p value of the t test

nAGQ = 1
plotPrefix <- "/../figures/"

##### Loading data ##### 
# Audio Data
audioData <- as.data.frame(read_parquet("../loc_data/df_gemaps_15s_end.parquet"))
audioData$type[audioData$DB == 'marloes'] = 'Read Aloud'
audioData$type[audioData$DB != 'marloes'] = 'GSST'

audioData$type <- factor(audioData$type, levels = c("Read Aloud", "GSST"))
names(audioData)[names(audioData) == "type"] <- "SpeechStyle" # Rename for clarification
audioData$ID  <- as.factor(audioData$ID)

# Speech features: Shimmer ######
formula <- 'shimmerLocaldB_sma3nz_amean ~ SpeechStyle + (1|ID)' # Declare formula

dataModel = audioData # Ensure correct data is taken

d0.1 <- lmer(formula,data=dataModel)

Anova(d0.1, type = 'III')

plot(effect("SpeechStyle", d0.1))

emmeans0.1 <- emmeans(d0.1, pairwise ~ SpeechStyle, adjust ="none", type = "response") #we don't adjust because we do this later
emm0.1 <- summary(emmeans0.1)$emmeans
emmeans0.1$contrasts

effSummary <- summary(eff_size(emmeans0.1, sigma=sigma(d0.1), edf=df.residual(d0.1)))
effSummary

# Speech features: Jitter ######
formula <- 'jitterLocal_sma3nz_amean ~ SpeechStyle + (1|ID)' # Declare formula

dataModel = audioData # Ensure correct data is taken

d0.1 <- lmer(formula,data=dataModel)

Anova(d0.1, type = 'III')

plot(effect("SpeechStyle", d0.1))

emmeans0.1 <- emmeans(d0.1, pairwise ~ SpeechStyle, adjust ="none", type = "response") #we don't adjust because we do this later
emm0.1 <- summary(emmeans0.1)$emmeans
emmeans0.1$contrasts

effSummary <- summary(eff_size(emmeans0.1, sigma=sigma(d0.1), edf=df.residual(d0.1)))
effSummary
