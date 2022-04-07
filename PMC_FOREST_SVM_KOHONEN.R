setwd("/Users/david/Desktop/davidtuto/pikachu")

library(magick)
library(EBImage)
library(caret)
library(e1071)
require(kohonen)
require(RColorBrewer)
library(keras)
library(ggplot2)     # to plot
library(gridExtra)   # to put more
library(grid) 
library(yardstick)
library(dplyr)

#for (j in paste0(path,i)){

path="/Users/david/Desktop/davidtuto1/"
data_dir <- file.path(dirname(path), "pikachu")



images<- NULL
df=data.frame()
for (i in list.files()){
  data <- image_read(i)
  data <- image_resize(data,'28x28')
  df<-data
  data <- as.numeric(data[[1]][1, ,])
  data <- data/255
  images <- rbind(images,data)
}
label<- rep("pikachu",times=length(list.files("/Users/david/Desktop/davidtuto1/pikachu")))
images1<- data.frame(images,label)
rownames(images1) <- 1:nrow(images1)


setwd("/Users/david/Desktop/davidtuto1/rondoudou")
images2 <- NULL
for (i in list.files()){
  data <- image_read(i)
  data <- image_extent(data,"40x40")
  data <- image_resize(data,'28x28')
  data <- as.numeric(data[[1]][1, ,])
  data <- data/255
  images2 <- rbind(images2,data)
}
label1<- rep("rondoudou",times=length(list.files("/Users/david/Desktop/davidtuto1/rondoudou")))
images3 <- data.frame(images2,label1)
rownames(images3) <- nrow(images1)+1:nrow(images3)+ nrow(images1)

setwd("/Users/david/Desktop/davidtuto1/Salameche")
images2 <- NULL
for (i in list.files()){
  data <- image_read(i)
  data <- image_extent(data,"40x40")
  data <- image_resize(data,'28x28')
  data <- as.numeric(data[[1]][1, ,])
  data <- data/255
  images2 <- rbind(images2,data)
}
label1<- rep("Salameche",times=length(list.files("/Users/david/Desktop/davidtuto1/Salameche")))
images4 <- data.frame(images2,label1)
rownames(images3) <- nrow(images3)+1:nrow(images4)+ nrow(images3)

setwd("/Users/david/Desktop/davidtuto1/Carapuce")
images2 <- NULL
for (i in list.files()){
  data <- image_read(i)
  data <- image_extent(data,"40x40")
  data <- image_resize(data,'28x28')
  data <- as.numeric(data[[1]][1, ,])
  data <- data/255
  images2 <- rbind(images2,data)
}
label1<- rep("Carapuce",times=length(list.files("/Users/david/Desktop/davidtuto1/Carapuce")))
images5 <- data.frame(images2,label1)
rownames(images5) <- nrow(images4)+1:nrow(images5)+ nrow(images4)


#------------------------------------------------------------------------------------------------

colnames(images1) <- colnames(images3)
colnames(images4) <- colnames(images3)
colnames(images5) <- colnames(images3)
final <- rbind(images1,images3,images4,images5)


### Spliter le dataframe

split <- sample(1:nrow(final),nrow(final)*0.7)
train <- final[split,]
test <- final[-split,]

####
library(neuralnet)

###

 
### model PMC
model<- neuralnet(label1~.,data=train, hidden=c(128,64,16), threshold=0.01,linear.output = FALSE)

#--------------------
output<- compute(model,test[,-785])
prediction <- output$net.result
#--------------------

y_pred <- predict(model,test[,-785])
  
prediction<-y_pred

yhat=data.frame("yhat"=ifelse(max.col(prediction[ ,1:4])==1, "Carapuce",
                              ifelse(max.col(prediction[ ,1:4])==2, "pikachu",
                              ifelse(max.col(prediction[ ,1:4])==4, "Salameche","rondoudou"))))

 cm= caret::confusionMatrix(as.factor(test[,785]), as.factor(yhat$yhat)) ### Toutes les stats
 
  conf_matrix(as.factor(test[,785]),as.factor(yhat$yhat))  #### Plot 



table(test$label1,apply(prediction,1,which.max))

print(cm)


### Model foret aléatoire 
train$label1 <- as.factor(train$label1)
library(randomForest)
model1 <- randomForest(label1~.,train,ntree=100,importance=TRUE,proximity=TRUE,mtry=56)

y_pred <- predict(model1,newdata=test[,-785])

confusionMatrix(as.factor(test$label),as.factor(y_pred))

conf_matrix(as.factor(test$label),as.factor(y_pred))

### Tuner le model
tuneRF(train[,-785],train[,785],plot=TRUE,ntreeTry = 100,stepFactor = 0.5)


### Donne un mtry=28

### Support vector Machine
model2 <- svm(factor(label1)~.,train,type="C-classification",na.action = na.omit)
y_pred <- predict(model2,newdata=test[,-785])

cm2<-confusionMatrix(as.factor(test[,785]),as.factor(y_pred))

conf_matrix(as.factor(test[,785]),as.factor(y_pred))


### Carte de KOHONEN
x <- scale(train[,1:784])
ujimatrix<- scale(test[,1:784])

som.r <- som(x, grid = somgrid(10, 11, "hexagonal"),toroidal = TRUE)
str(som.r)
plot(som.r, type = "mapping")
#plot(som.r, type = "mapping", pchs = 20, main = "Mapping Type SOM")
plot(som.r, main = "Default SOM Plot")
plot(som.r, type = "dist.neighbours", palette.name = terrain.colors)  ### Mapping Distance
coolBlueHotRed <- function(n, alpha = 1) {
  rainbow(n, end=4/6, alpha=alpha)[n:1]
}
plot(som.r, type="codes", codeRendering="segments", main="Profil des vecteurs référents", palette.name=coolBlueHotRed)
text(som.r$grid$pts, labels = som.r$unit.classif, cex = 1.5)

som.prediction <- predict(som.r, newdata = ujimatrix,
                          trainX = x,
                          trainY = factor(train[,785]))

table(test[,785], som.prediction$prediction)

conf_matrix(test[,785],som.prediction$prediction)


kohmap <- xyf(x, classvec2classmat(train[,785]),
              grid = somgrid(5, 5, "hexagonal"), rlen=100)
plot(kohmap, type="changes")
l<-plot(kohmap, type="codes", main = c("Codes X", "Codes Y"))

plot(kohmap, type="counts")

plot(kohmap, type="quality", palette.name = coolBlueHotRed)
plot(kohmap, type="mapping", 
     labels = train[,785], col = 2+1,
     main = "mapping plot")

xyfpredictions <- classmat2classvec(predict(kohmap)$unit.predictions)
bgcols <- c("gray", "pink", "lightgreen")
plot(kohmap, type="mapping", col = 2+1,
     pchs = 2, bgcol = bgcols[as.integer(xyfpredictions)], 
     main = "another mapping plot")


n<- table(som.r$unit.classif)
print(n)


###### Fonction pour ploter les matrices de confuqion


conf_matrix <- function(df.true, df.pred, title = "", true.lab ="True Class", pred.lab ="Predicted Class",
                        high.col = 'red', low.col = 'white') {
  #convert input vector to factors, and ensure they have the same levels
  df.true <- as.factor(df.true)
  df.pred <- factor(df.pred, levels = levels(df.true))
  
  #generate confusion matrix, and confusion matrix as a pecentage of each true class (to be used for color) 
  df.cm <- table(True = df.true, Pred = df.pred)
  df.cm.col <- df.cm / rowSums(df.cm)
  
  #convert confusion matrices to tables, and binding them together
  df.table <- reshape2::melt(df.cm)
  df.table.col <- reshape2::melt(df.cm.col)
  df.table <- left_join(df.table, df.table.col, by =c("True", "Pred"))
  
  #calculate accuracy and class accuracy
  acc.vector <- c(diag(df.cm)) / c(rowSums(df.cm))
  class.acc <- data.frame(Pred = "Class Acc.", True = names(acc.vector), value = acc.vector)
  acc <- sum(diag(df.cm)) / sum(df.cm)
  
  #plot
  ggplot() +
    geom_tile(aes(x=Pred, y=True, fill=value.y),
              data=df.table, size=0.2, color=grey(0.5)) +
    geom_tile(aes(x=Pred, y=True),
              data=df.table[df.table$True==df.table$Pred, ], size=1, color="black", fill = 'transparent') +
    scale_x_discrete(position = "top",  limits = c(levels(df.table$Pred), "Class Acc.")) +
    scale_y_discrete(limits = rev(unique(levels(df.table$Pred)))) +
    labs(x=pred.lab, y=true.lab, fill=NULL,
         title= paste0(title, "\nAccuracy ", round(100*acc, 1), "%")) +
    geom_text(aes(x=Pred, y=True, label=value.x),
              data=df.table, size=4, colour="black") +
    geom_text(data = class.acc, aes(Pred, True, label = paste0(round(100*value), "%"))) +
    scale_fill_gradient(low=low.col, high=high.col, labels = scales::percent,
                        limits = c(0,1), breaks = c(0,0.5,1)) +
    guides(size=F) +
    theme_bw() +
    theme(panel.border = element_blank(), legend.position = "bottom",
          axis.text = element_text(color='black'), axis.ticks = element_blank(),
          panel.grid = element_blank(), axis.text.x.top = element_text(angle = 30, vjust = 0, hjust = 0)) +
    coord_fixed()
  
} 




##### RBM
modelRBM <- RBM(x =as.matrix(train[,-785]),n.iter = 1000, n.hidden = 100, size.minibatch = 10)

RBM(x = train[,-785], plot = TRUE, n.iter = 1000, n.hidden = 30, size.minibatch = 10)

test <- as.matrix(test[,-785])

# Reconstruct the image with modelRBM
g<- as.matrix(train[,-785])
ReconstructRBM(test = g[1,], model = modelRBM)


------
  

PredictRBM(test = , labels =as.numeric(as.factor(test$label1))-1, model = modelRBM)


library(data.table)
d<- t(train)
colnames<-rownames(d)
rownames<-colnames(d)

x<- as.numeric(d[-785,])

x<-as.matrix(x,nrow=784)
