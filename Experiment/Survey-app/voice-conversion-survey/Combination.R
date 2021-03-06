library(stringr)
combination <- function(){
  X = c()
  for (i in c(1:2)){
    
    for (j in c(1:2)){
      
      for (k in c(1:4)){
        X <- rbind( X , (c(i,j,k)))
        
      }
    
      
    }
    for (l in c(3:4)){
      X <- rbind(X , c(i,l,1))
    }
    
  }
  for (j in c(5:6)){
    for (i in c(5:8)){
      X <- rbind(X , c(3,j,i))
    }}
  
  return(X)
}


combination2 <- function(){
  Y <- c()
  for (i in c(1:2)){
    
    for (j in c(1:2)){
      
      for (k in c(1:4)){
        Y <- rbind(Y, c(i,j,k))
        
      }
    }
  }
  return(Y)
  
  
}
# 
# models = c("AutoVC", "StarGAN", "Baseline")
# categories = c("Danish_Danish", "English_English", "20min", "10min", "WaveRNN", "WORLD")
# subcategories = c("Male_Male", "Female_Female", "Male_Female", "Female_Female", "Male_English", "Male_Danish", "Female_English", "Female_Danish")
# 
# voices = c("source", "target", "converted")
# 
# X <- combination()
# Y <- combination2()
# X
# Y
# 
# 
# SamplesA <- sample(nrow(Y), nrow(Y), replace = F)
# SamplesB <- sample(nrow(X), nrow(X), replace = F)

get_wavs_experiment <- function(model, task, subtask, q, seed){
    set.seed(seed)
 
    if (q == "real_fake"){
      list_of_fakes <- list.files(sprintf("www/%s/%s/%s", model, task, subtask))
      fake <- sample(list_of_fakes, 1, F)
      t <- task
      s <- strsplit(subtask[1], "_")[[1]]
      
      baseline_category <- paste(s[2],t, sep = "_")
      
      if (model == "AutoVC"){
        baseline = "WaveRNN"
      }
      else if (model == "StarGAN"){
        baseline = "WORLD"
      }
    
      
      list_of_reals <- list.files(paste("www", "Baseline", baseline, baseline_category, sep = "/"))
      
      name <- sample(list_of_reals, 1)
      
      real<-sample(list.files(paste("www", "Baseline", baseline, baseline_category, name, sep = "/")),1)
      
      path_to_fake <- paste(model, task, subtask, fake, sep= "/")
      path_to_real <- paste("Baseline", baseline, baseline_category, name, real, sep = "/")
      # print(c(path_to_real, path_to_fake))
      return(c(path_to_real, path_to_fake))
    }
  
  
  
  
    
    else if (q == "similarity" & model != "Baseline"){
      
      list_of_fakes <- list.files(paste("www", model, task, subtask, sep = "/"))
      
      fakes <- sample(list_of_fakes, 2, F)
      if (model == "AutoVC"){
        split <- strsplit(str_remove(fakes[1], ".wav"), "_")[[1]]
        # print(split)
        from_and_to <- paste(split[1], split[length(split)], sep = "_")
      }
      else if (model == "StarGAN"){
        split <- strsplit(str_remove(fakes[1], ".wav"), "-")[[1]]
        a <- strsplit(split[1], "_")[[1]]
        b <- split[length(split)]
        from_and_to <- paste(a[1], b, sep = "_")
        # print(a)
        # print(b)
      }
    
      name <- split[length(split)]
      
      
      print(from_and_to)
      
      
      if (model == "AutoVC"){
        baseline = "WaveRNN"
      }
      else if (model == "StarGAN"){
        baseline = "WORLD"
      }
      # print(name)
      list_of_reals <- list.files(paste("www", "Baseline", baseline, "persons", name, sep = "/"))
      # print(list_of_reals)
      real <- sample(list_of_reals, 1)
      
      
      path_to_fake1 <- paste(model, task, subtask, fakes[1], sep= "/")
      path_to_fake2 <- paste(model, task, subtask, fakes[2], sep= "/")
      path_to_real <- paste("Baseline", baseline, "persons", name, real, sep = "/" )
      # print(c(path_to_fake1, path_to_real, path_to_fake2, name))
      return(c(path_to_fake1, path_to_real, path_to_fake2, from_and_to))
    }
  
  
  
  
  
  else if  (q == "similarity" & model == "Baseline"){
    names <- sample(list.files(paste("www", model, task, subtask, sep = "/")),2, replace = F)
    list_of_reals1 <- list.files(paste("www", model, task, subtask, names[1], sep = "/"))
    list_of_reals2 <- list.files(paste("www", model, task, subtask, names[2], sep = "/"))
    
    person1 <- sample(list_of_reals1, 2, F)
    person2 <- sample(list_of_reals2, 1, F)
    
    path_to_fake1 <- paste(model, task, subtask, names[1], person1[1], sep= "/")
    path_to_fake2 <- paste(model, task, subtask, names[2], person2, sep= "/")
    path_to_real <- paste(model, task, subtask, names[1], person1[2], sep = "/" )
    # print(c(path_to_fake1, path_to_real, path_to_fake2, names[2]))
    return(c(path_to_fake1, path_to_real, path_to_fake2, names[2]))

  }
    
}


# get_wavs_experiment("StarGAN", "English", "Male_Female", "similarity", 1)
