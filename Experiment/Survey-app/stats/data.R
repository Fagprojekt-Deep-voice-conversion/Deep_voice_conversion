library(googlesheets4)
library(googledrive)
options(gargle_oauth_cache = ".secrets")
library(lubridate)
library(magrittr)

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


S$Time <- ymd_hms(S$Time, tz = "UTC") %>% with_tz(tzone = "Europe/Copenhagen")
Q$Time <- ymd_hms(Q$Time, tz = "UTC") %>% with_tz(tzone = "Europe/Copenhagen")
Fool$Time <- ymd_hms(Fool$Time, tz = "UTC") %>% with_tz(tzone = "Europe/Copenhagen")
P$Time <- ymd_hms(P$Time, tz = "UTC") %>% with_tz(tzone = "Europe/Copenhagen")
PAV$Time <- ymd_hms(PAV$Time, tz = "UTC") %>% with_tz(tzone = "Europe/Copenhagen")