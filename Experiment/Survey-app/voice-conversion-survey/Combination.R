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
    for (i in c(5:8)){
      X <- rbind(X , c(3,6,i))
    }
  
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

models = c("AutoVC", "StarGAN", "Baseline")
categories = c("Danish_Danish", "English_English", "20min", "10min", "Baseline")
subcategories = c("Male_Male", "Female_Female", "Male_Female", "Female_Female", "Male_English", "Male_Danish", "Female_English", "Female_Danish")

voices = c("source", "target", "converted")

X <- combination()
Y <- combination2()
X
Y


SamplesA <- sample(nrow(Y), nrow(Y), replace = F)
SamplesB <- sample(nrow(X), nrow(X), replace = F)
X[1,]
