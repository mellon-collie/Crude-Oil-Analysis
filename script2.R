library(dplyr)
data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/PrudhviData/final_oil.csv")
data
data <- data[rev(rownames(data)),]
data
dates_num <- as.character(data$Date)
n_dates
for(i in 1:nrow(data)) {
  splits <- strsplit(dates_num[i]," ")
  print(splits)
  date <- paste(strsplit(splits[[1]][2],",")[1],"/",sep="")
  print(date)
  
  if(splits[[1]][1] == "Jan") {
    date <- paste(date,"01/",sep="")
    
  } else if(splits[[1]][1] == "Feb") {
    date <- paste(date,"02/",sep="")
    
  } else if(splits[[1]][1] == "Mar") {
    date <- paste(date,"03/",sep="")
    
  } else if(splits[[1]][1] == "Apr") {
    date <- paste(date,"04/",sep="")
  } else if(splits[[1]][1] == "May") {
    date <- paste(date,"05/",sep="")
    
  } else if(splits[[1]][1] == "Jun") {
    date <- paste(date,"06/",sep="")
    
  } else if(splits[[1]][1] == "Jul") {
    date <- paste(date,"07/",sep="")
    
  } else if(splits[[1]][1] == "Aug") {
    date <- paste(date,"08/",sep="")
    
  } else if(splits[[1]][1] == "Sep") {
    date <- paste(date,"09/",sep="")
    
  } else if(splits[[1]][1] == "Oct") {
    date <- paste(date,"10/",sep="")
    
  } else if(splits[[1]][1] == "Nov") {
    date <- paste(date,"11/",sep="")
    
  } else if(splits[[1]][1] == "Dec") {
    date <- paste(date,"12/",sep="")
    
  }
  date <- paste(date,substr(splits[[1]][3],3,4),sep="")
  dates_num[i] = date
}
data <- cbind(data,as.data.frame(dates_num))
data
data <- data[,2:11]
data
names(data) <- c("X","US.Price","Dollar_eq","OilPrice","OilOpen","OilHigh","OilLow","OilVol","OilChange.","dates_num")
data_2 <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/dates.csv")
final<- merge(x=data_2,y=data,by="dates_num",all.x = TRUE)
final <- final[order(as.Date(final$dates_num,format="%d/%m/%Y")),]
final
final <- final[,2:10]

final_data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final9.csv")
final
final_data <- cbind(final_data,final)
final_data
names(final_data)
final
write.csv(final_data,file="/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final10.csv",row.names = F)
