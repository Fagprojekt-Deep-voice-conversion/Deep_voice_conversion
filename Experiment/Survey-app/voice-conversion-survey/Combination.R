combination <- function(){
  X = c()
  for (i in c(1:2)){
    
    for (j in c(1:7)){
      
      for (k in c(1:4)){
        X <- rbind( X , (c(i,j,k)))
        
      }
    }
    for (i in c(5:8)){
      X <- rbind(X , c(3,8,i))
    }
  }
  return(X)
}