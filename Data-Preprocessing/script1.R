data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/PrudhviData/tiger_logistics.csv")
data_2 <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/dates.csv")
final_data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final8.csv")
names(final_data)
dates_num <- character()
for(i in 1:nrow(data)) {
  year <- strsplit(strsplit(as.character(data[i,1]),split='-', fixed=TRUE)[[1]][1],split='0',fixed=TRUE)[[1]][2]
  month <- strsplit(as.character(data[i,1]),split='-', fixed=TRUE)[[1]][2]
  day <- strsplit(as.character(data[i,1]),split='-', fixed=TRUE)[[1]][3]
  dates_num[i]<- as.character(paste(day,month,year,sep="/"))
}

data <- cbind(data,as.data.frame(dates_num))
data <- data[,2:ncol(data)]
names(data)
final<- merge(x=data_2,y=data,by="dates_num",all.x = TRUE)
final <- final[order(as.Date(final$dates_num,format="%d/%m/%Y")),]
final <- final[,c(2:6)]
names(final)
names(final) <- c("TigerOpen","TigerHigh","TigerLow","TigerClose","TigerAdjClose")
names(final)
final_data <- cbind(final_data,final)
write.csv(final_data,file="/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final12.csv",row.names = F)
