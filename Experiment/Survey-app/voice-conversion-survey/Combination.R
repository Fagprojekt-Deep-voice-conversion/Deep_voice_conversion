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
X <- combination()
X
Y <- combination2()
Y
