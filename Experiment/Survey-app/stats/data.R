library(googlesheets4)
library(googledrive)
options(gargle_oauth_cache = ".secrets")
library(lubridate)
library(magrittr)
library(dplyr)
library(tidyr)

# Get data
drive_auth(cache = ".secrets", email = "luke.leindance@gmail.com")
gs4_auth(token = drive_token())
ss = "https://docs.google.com/spreadsheets/d/1Y2Hu04dY-chxSPdVgcUefXSTs6zvG6lkzAFlWhICPJA/edit#gid=0"

S <- read_sheet(ss, sheet = "Similarity")
Q <- read_sheet(ss, sheet = "Quality")
Fool <- read_sheet(ss, sheet = "Fakeness")
P <- read_sheet(ss, sheet = "Persons")
PAV <- read_sheet(ss, sheet = "ConversionsAV")
PSG <- read_sheet(ss, sheet = "ConversionsSG")


# data prep 
n_participants <- nrow(S)

transform_data <- function(data){
  data %>% 
    mutate(S30M = SDMM) %>% 
    mutate(A30M = ADMM) %>% 
    gather(key = "conv_type", value = "score", -c(Time, Zone, Age, Gender)) %>%
    mutate(model = stringr::str_extract(conv_type, "^.{1}")) %>%
    mutate(model = ifelse(model == "A", "AutoVC", ifelse(model == "S", "StarGAN", "Baseline"))) %>%
    mutate(conv_type = gsub("^.{1}", "", conv_type)) %>%
    mutate(score = as.integer(score)) %>%
    mutate(Age = as.integer(Age)) %>%
    mutate(Time = ymd_hms(Time, tz = "UTC") %>% with_tz(tzone = "Europe/Copenhagen")) %>% 
    mutate(vocoder = ifelse(model == "AutoVC", "WaveRNN", "World")) %>% 
    mutate(vocoder = ifelse((model == "Baseline") & (stringr::str_extract(conv_type, "^.{4}") == "Wave"), "WaveRNN", "World")) %>% 
    mutate(model = ifelse(model == "Baseline", ifelse(vocoder == "WaveRNN", "WaveRNN Baseline", "World Baseline"), model)) %>% 
    mutate(conv_type =  gsub("Wave", "", conv_type)) %>% 
    mutate(conv_type = gsub("World", "", conv_type)) %>% 
    mutate(experiment = ifelse(conv_type %in% c("10M", "20M", "30M"), "Amount", "Type")) %>% 
    mutate(conv_type = ifelse(conv_type == "10M", "10 min", ifelse(conv_type == "20M", "20 min", ifelse(conv_type == "30M", "30 min", conv_type))))
}

S <- transform_data(S)

Q <- transform_data(Q)

Fool <- transform_data(Fool) %>% filter(conv_type != "30 min")
