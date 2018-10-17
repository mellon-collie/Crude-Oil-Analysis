data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/Sept-30/USExportstoIndiaMonthly.csv")
data <- data[rev(rownames(data)),]
year_month <- data$B
value <- data$D
dates_num <- character()
for(i in 1:length(year_month)) {
  year <-  substr(year_month[i],3,4)
  print(year)
  month <- substr(year_month[i],5,7)
  print(month)
  dates_num[i] <- paste("01",month,sep="/")
  dates_num[i] <- paste(dates_num[i],year,sep="/")
}

data <- cbind(as.data.frame(dates_num),as.data.frame(value))
data_2 <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/dates.csv")
final_data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final6.csv")
final<- merge(x=data_2,y=data,by="dates_num",all.x = TRUE)
final <- final[order(as.Date(final$dates_num,format="%d/%m/%Y")),]
final <- as.data.frame(final[order(as.Date(final$dates_num,format="%d/%m/%Y")),2])
names(final) <- c("UsExportsIndia")
final_data <- cbind(final_data,final)
write.csv(final_data,file="/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final7.csv",row.names = F)
