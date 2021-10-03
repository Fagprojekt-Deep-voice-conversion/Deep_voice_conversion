library(googlesheets4)
library(googledrive)
source("Combination.R")
options(gargle_oauth_cache = ".secrets")
library(ggplot2)

drive_auth(cache = ".secrets", email = "peter@groenning.net")
gs4_auth(token = drive_token())
ss = "https://docs.google.com/spreadsheets/d/1Y2Hu04dY-chxSPdVgcUefXSTs6zvG6lkzAFlWhICPJA/edit#gid=0"

S <- read_sheet(ss, sheet = "Similarity")
Q <- read_sheet(ss, sheet = "Quality")
Fool <- read_sheet(ss, sheet = "Fakeness")
P <- read_sheet(ss, sheet = "Persons")
PAV <- read_sheet(ss, sheet = "ConversionsAV")
PSG <- read_sheet(ss, sheet = "ConversionsSG")
#### SIMILARITY ####
data <- S[,-c(1,2,3,4)]
i <- c(1:28)
data[ , i] <- apply(data[ , i], 2,            # Specify own function within apply
                    function(x) as.numeric(as.character(x)))
MOS <- colMeans(data) 
X <- t(cbind(MOS[c(1:8)], MOS[c(11:18)]))
colnames(X) <- c("DMM", "DFF", "DMF", "DFM", "EMM", "EFF", "EMF", "EFM")
barplot(X, beside = T, legend.text = c("AutoVC", "StarGAN"), col = c("skyblue", "orange"), main = "Similarity")
# X <- X/5 * (5-1) +1
barplot(X, beside = T, legend.text = c("AutoVC", "StarGAN"), col = c("skyblue", "orange"), main = "Similarity")
boxplot(data)
sem <- apply(data, 2, sd, na.rm = T) / length(data$ADMM)
sem
a = 0.05
MOS + pt(a/2, length(data$A10M) - 1) * sem
MOS - pt(a/2, length(data$A10M) - 1) * sem

#### QUALITY ####
data <- Q[,-c(1,2,3,4)]
i <- c(1:28)
data[ , i] <- apply(data[ , i], 2,            # Specify own function within apply
                    function(x) as.numeric(as.character(x)))
MOS <- colMeans(data) 
X <- t(cbind(MOS[c(1:8)], MOS[c(11:18)]))
colnames(X) <- c("DMM", "DFF", "DMF", "DFM", "EMM", "EFF", "EMF", "EFM")
barplot(X, beside = T, legend.text = c("AutoVC", "StarGAN"), col = c("skyblue", "orange"), main = "Quality")
X <- X/5 * (5-1) +1
barplot(X, beside = T, legend.text = c("AutoVC", "StarGAN"), col = c("skyblue", "orange"), main = "Quality")
### Fool Score ### 
data <- Fool[,-c(1,2,3,4)]
data
i <- c(1:16)
data[ , i] <- apply(data[ , i], 2,            # Specify own function within apply
                    function(x) as.numeric(as.character(x)))
MOS <- 1-colMeans(data) 
X <- t(cbind(MOS[c(1:8)], MOS[c(11:18)]))
colnames(X) <- c("DMM", "DFF", "DMF", "DFM", "EMM", "EFF", "EMF", "EFM")
barplot(X, beside = T, legend.text = c("AutoVC", "StarGAN"), col = c("skyblue", "orange"), main = "Fool Score")




#### BASELINE SOUND QUALITY ####
data <- P[,-c(1,2,3,4)]
data
i <- c(1:16)
data[ , i] <- apply(data[ , i], 2,            # Specify own function within apply
                    function(x) as.numeric(as.character(x)))

MOS <- colMeans(data, na.rm = T) 
X <- t(cbind(MOS[c(1:8)], MOS[c(9:16)]))

colnames(X) <- c("Obama", "Trump", "Hillary", "Michelle", "Mette", "Helle", "Lars", "Anders")
barplot(X, beside = T, legend.text = c("WaveRNN", "WORLD"), col = c("skyblue", "orange"), main = "Baseline Sound Quality / Naturalness")



#### CONVERSION AUTOVC SOUND QUALITY ####
data <- PAV[,-c(1,2,3,4)]
data
i <- c(1:24)
data[ , i] <- apply(data[ , i], 2,            # Specify own function within apply
                    function(x) as.numeric(as.character(x)))
data

MOS <- colMeans(data, na.rm = T)


Y <- matrix(t(MOS), nrow = 3, ncol = 8)
X
# X <- t(cbind(MOS[c(1:8)], MOS[c(9:16)]))

colnames(Y) <- c("Obama", "Trump", "Hillary", "Michelle", "Mette", "Helle", "Lars", "Anders")
barplot(colMeans(Y), beside = T,  col = c("skyblue", "orange"), main = "Conversion AutoVC Sound Quality / Naturalness")




data <- PSG[,-c(1,2,3,4)]
data
i <- c(1:24)
data[ , i] <- apply(data[ , i], 2,            # Specify own function within apply
                    function(x) as.numeric(as.character(x)))
data

MOS <- colMeans(data, na.rm = T)


X <- matrix(t(MOS), nrow = 3, ncol = 8)
X
# X <- t(cbind(MOS[c(1:8)], MOS[c(9:16)]))

colnames(X) <- c("Obama", "Trump", "Hillary", "Michelle", "Mette", "Helle", "Lars", "Anders")
barplot(colMeans(X), beside = T, col = c("skyblue", "orange"), main = "Conversion AutoVC Sound Quality / Naturalness")



barplot(rbind(colMeans(Y), colMeans(X)), beside = T, legend.text = c("AutoVC", "StarGAN"), col = c("skyblue", "orange"), main = "Mean Conversion Sound Quality / Naturalness")


x = 0.25 * (5-1)
0.25 *(5-1) +1

