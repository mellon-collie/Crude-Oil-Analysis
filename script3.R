data <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/Sept-30/USExportsOverallMonthly (copy).csv")

dates <- character()
values <- character()

c <- 0

for(i in 1:nrow(data)) {
  year <- strsplit(strsplit(as.character(data[i,1]),split="-",fixed=TRUE)[[1]][1],split="0",fixed=TRUE)[[1]][2]
  j <- 2
  repeat{
    if(j%%2 == 0 && j<=11) {
      month <- strsplit(as.character(data[i,j]),split="/",fixed=TRUE)[[1]][1]
      date <- strsplit(as.character(data[i,j]),split="/",fixed=TRUE)[[1]][2]
      date <-  substr(date,1,nchar(date)-1)
      if(!is.na(date)) {
        dates[c] <- paste(date,"/",month,sep="")
        dates[c] <- paste(dates[c],"/",year,sep="")
      } 
      
      j <- j + 1
    } else if(j%%2 == 1 && j<=11) {
      value <- as.character(data[i,j])
      if(!is.na(as.numeric(value))) {
        values[c] <- value
        c <- c+1
      }
      j <- j + 1
      
      
      
    } else {
      break
    }
  }
}
dates <- as.data.frame(dates)
values <- as.data.frame(values)
final <- cbind(dates,values)
names(final) <- c("dates_num","UsExportsOverall")
data_2 <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/dates.csv")
final<- merge(x=data_2,y=final,by="dates_num",all.x = TRUE)
final <- as.data.frame(final[order(as.Date(final$dates_num,format="%d/%m/%Y")),2])
names(final) <- c("UsExportsOverall")
final_dataset <- read.csv("/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final5.csv")
final_dataset <- cbind(final_dataset,final)
write.csv(final_dataset,file="/home/nishant/Documents/sem5/DA/Crude Oil/FinalDataSet/final6.csv",row.names = F)