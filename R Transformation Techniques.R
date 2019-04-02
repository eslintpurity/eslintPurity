library(plyr)
library(stringr)
options(stringsAsFactors = FALSE)


color <- c(blue, black, blue, blue, black, green, yellow, green, green, blue)
value <- seq(1:10)

##slicing and dicing
subset(df, color == "blue")
transform(df, double = 2 * value)
summarise(df, double = 2 * value)
summarise(df, total = sum(value))
arrange(df, color)

What is plyr? It's a bundle of awesomeness (i.e. an R package) that makes it simple
to split apart data, do stu to it, and mash it back together. This is a common data
manipulation step.
Or, from the documentation:
\plyr is a set of tools that solves a common set of problems: you need to break a
big problem down into manageable pieces, operate on each pieces and then put all
the pieces back together. It's already possible to do this with split and the apply
functions, but plyr just makes it all a bit easier. . . "
This is a very quick introduction to plyr. For more details have a look at the plyr
site: http://had.co.nz/plyr/ and particularly Hadley Wickham's introductory
guide The split-apply-combine strategy for data analysis.
http://had.co.nz/plyr/plyr-intro-090510.pdf