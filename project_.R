#installing recount
if(!require('recount', quietly = TRUE)){
  #note: use 'http' if 'https' is not supported
  source(file = "http://bioconductor.org/biocLite.R")
  biocLite("recount")
  library(recount)
}

download_study('SRP018008')
