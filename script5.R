library(gdata)
data <- read.xls("/home/nishant/Documents/sem5/DA/Crude Oil/Sept-30/Table_11.1a_World_Crude_Oil_Production___OPEC_Members (copy).xlsx")
dates_num <- character()
for(i in 1:nrow(data)) {
  year <- substr(strsplit(as.character(data[i,1]),split = " ",fixed=TRUE)[[1]][1],3,4) 
  month <- strsplit(as.character(data[i,1]),split = " ",fixed=TRUE)[[1]][2]
  print(year)
  if(month == "January") {
    month <- "01"
  } else if(month == "February") {
    month <- "02"
  } else if(month == "February") {
    month <- "02"
  } else if(month == "March") {
    month <- "03"
  } else if(month == "April") {
    month <- "04"
  } else if(month == "May") {
    month <- "05"
  } else if(month == "June") {
    month <- "06"
  } else if(month == "July") {
    month <- "07"
  } else if(month == "August") {
    month <- "08"
  } else if(month == "September") {
    month <- "09"
  } else if(month == "October") {
    month <- "10"
  } else if(month == "November") {
    month <- "11"
  } else if(month == "December") {
    month <- "12"
  }
  dates_num[i] <- paste("01",month,sep="/")
  dates_num[i] <- paste(dates_num[i],year,sep="/")
}
data <- cbind(data,as.data.frame(dates_num))
data <- data[,2:ncol(data)]
data_2 <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/dates.csv")
final_data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final7.csv")
final<- merge(x=data_2,y=data,by="dates_num",all.x = TRUE)
final <- final[order(as.Date(final$dates_num,format="%d/%m/%Y")),]
ncol(final)
final <- as.data.frame(final[order(as.Date(final$dates_num,format="%d/%m/%Y")),c(2:15)])
final_data <- cbind(final_data,final)
write.csv(final_data,file="/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final8.csv",row.names = F)
