# chooseVars selects those variables which are most informative as determined by nearZeroVar,
# and performs cross-validation with the rfcv() function to estimate error:
chooseVars <- function() {
     library(caret); library(randomForest)
     
     fullset <- read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings=c("","#DIV/0!", "NA" ))   #Load the entire training set
     trset <- fullset[,8:159]             #Eliminate extraneous data related to data collection
     
     #Keep only the most informative variables; save as "finaltr":
     nzv <- nearZeroVar(trset, saveMetrics=TRUE)#, freqCut=1, uniqueCut=3, saveMetrics=TRUE)
     finaltr <- subset(trset,,select=(nzv$nzv==FALSE))
     # Eliminate the columns that contain NAs
     nalist <- sapply(finaltr, function(n) sum(is.na(n)) )
     finaltr <- finaltr[,nalist==0]

     #Add the response of interest to the set, with column name "classe"
     finaltr <- cbind(fullset$classe, finaltr)
     names(finaltr)[1] <- "classe"
     
#      set.seed(4700)
#      cvtest <- rfcv(finaltr[,-1], finaltr[,1], cv.fold=8)
#      plot(cvtest$n.var, cvtest$error.cv, pch=19, log="x")
     

}


buildModel <- function(filetoprocess, chosenVars) {
     
     teset <- read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings=c("","#DIV/0!", "NA" ))
     finalte <- teset[, colnames(teset) %in% colnames(finaltr) ]
     rffit <- randomForest(finaltr[,-1], as.factor(finaltr$classe) )
     pred = predict(rffit,newdata=finalte)
}
     
     

#namelist <- chooseVars("pml-training.csv")
#rfmodel <- buildModel("pml-testing.csv", namelist)

