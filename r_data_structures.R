#################################### DATA STRUCTURES ###################################################

##################################### VECTORS ############################################

# 1. NUMERIC VECTORS
numVect <- c(1, 4, 6, 88, 45, 7, 23)

# Checking the type, indexing, length and summation operations
is(numVect)
numVect[2]
numVect[3:6]
length(numVect)
sum(numVect)

# 2. CHARACTER VECTORS
charVect <- c("Nathasa", "Alexandra", "Amanda", "Livingstone", "Sandra", "Nightingale")

# Checking the type, indexing, length and summation operations
is(charVect)
charVect[2]
charVect[3:6]
length(charVect)


###################### FACTORS #################################

citizen <- factor(c("uk", "us", "no", "au", "uk", "us", "us"))
citizen
unclass(citizen)
citizen[5:7]
citizen[5:7, drop = TRUE]

