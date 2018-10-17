data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/PrudhviData/final_import.csv")
data <- data[,2:6]
data
data_2 <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/dates.csv")
final_data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final10.csv")
final<- merge(x=data_2,y=data,by="dates_num",all.x = TRUE)
final
final
final <- final[order(as.Date(final$dates_num,format="%d/%m/%Y")),]
final <- final[,2:5]
final_data <- cbind(final_data,final)
final_data
write.csv(final_data,file="/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final11.csv",row.names = F)