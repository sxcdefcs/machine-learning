##随机森林代码
#设置工作空间
getwd()
setwd("D:/")
getwd()
library(ggpol)
#准备数据
data<-read.table("D:/文件.csv",sep=",",header = TRUE)
data<-data[-1]
data$class<-factor(data$class,levels=c(1,0),labels = c("case","others"))
head(data)
dim(data)
table(data$class) 
#划分数据集
library(caret)
set.seed(4232)
trainIndex <- createDataPartition(data$class, p = .5, 
                                  list = FALSE, 
                                  times = 1)

head(trainIndex)
Train<-data[trainIndex,]
head(Train)
Test<-data[-trainIndex,]
head(Test)
#简单随机森林，寻找最优ntree
library(randomForest)
set.seed(123)
rf0<-randomForest(class~.,data=Train,ntree=1000)
rf0
plot(rf0)
pdf(file='RF-1-ntree.pdf', height=6,width=10, family='GB1')
plot(rf0)
dev.off()
t<-800
#寻找最优mtry
n<-length(names(Train))
n
err<-as.numeric()
set.seed(1)
min<-as.numeric(100)
num<-0
for (i in 1:(n-1)){
  rf1<- randomForest(class~., data=Train, mtry=i,ntree=t)
  print(rf1$err.rate)
  err[i]<-mean(rf1$err.rate[,1])
  print(err[i])
  if(err[i]<min) {    
    min=err[i]     
    num=i }
}
print(min)
print(num)
mtry<-num
plot(1:(n-1),err,type="b",xlab = "mtry",main="OOB Errors")
abline(v=mtry,lty=2)
pdf(file='RF-2-mtry.pdf', height=6,width=10, family='GB1')
plot(1:(n-1),err,type="b",xlab = "mtry",main="OOB Errors")
abline(v=mtry,lty=2)
dev.off()
#构建正式模型
set.seed(7)
rf<-randomForest(class~.,data = Train,mtry=mtry,ntree=t,importance=T)
rf
plot(rf,main="Bagging OOB Errors")
legend("topright", colnames(rf$err.rate),lty=1:3,col=c(4,1,1))
pdf(file='RF-3-rf.pdf', height=6,width=10, family='GB1')
plot(rf,main="Bagging OOB Errors")
legend("topright", colnames(rf$err.rate),lty=1:3,col=c(4,1,1))
dev.off()
#输出重要性
a1<-data.frame(importance(rf))
write.csv(a1,"MDD-rf.csv")
importance(rf)
varImpPlot(rf,n.var=10)
pdf(file='RF-4-gn.pdf', height=6,width=10, family='GB1')
varImpPlot(rf,n.var=10)
dev.off()
pdf(file='RF-5-xvar.pdf', height=6,width=10, family='GB1')
partialPlot(rf,Train,x.var=hsa.miR.22.3p)
partialPlot(rf,Train,x.var=hsa.let.7a.5p)
partialPlot(rf,Train,x.var=hsa.miR.25.3p)
dev.off()
#模型预测
pre<-predict(rf,newdata=Test,type = "class")
pre
#模型检验
library(caret)
rf.cf<-caret::confusionMatrix(as.factor(pre),as.factor(Test$class),dnn = c('Actual', 'Predicted'))
rf.cf
ggplot() + geom_confmat(aes(x =Test$class, y = pre),normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction")+
  scale_fill_gradient2(low="white", high="lightblue")+theme_minimal()+theme_bw()
pdf(file='rf-3-hx.pdf', height=6,width=6, family='GB1')
ggplot() + geom_confmat(aes(x =Test$class, y = pre),normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction")+
  scale_fill_gradient2(low="white", high="lightblue")+theme_minimal()+theme_bw()
dev.off()
#绘制ROC曲线
pre<-predict(rf,newdata=Test,type = "prob")
pre
library(pROC)
roc.rf <- roc(response=Test$class,predictor=as.numeric(pre[,1]))
roc.rf 
plot(roc.rf, print.auc=TRUE,auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("lightblue", "lightblue"),max.auc.polygon=TRUE,auc.polygon.col="lightblue",max.auc.polygon.col="white",print.thres=TRUE)
pdf(file='rf-2-roc.pdf', height=6,width=6, family='GB1')
plot(roc.rf, print.auc=TRUE,auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("lightblue", "lightblue"),max.auc.polygon=TRUE,auc.polygon.col="lightblue",max.auc.polygon.col="white",print.thres=TRUE)
dev.off()

##xgboost代码
#设置工作空间
getwd()
setwd("D:/")
getwd()
#准备数据
library(caret)
library(xgboost)
library(Matrix)
data<-read.table("D:/文件.csv",sep=",",header = TRUE)
data<-data[-1]
head(data)
data$class<-factor(data$class,levels=c(1,0),labels = c("case","control"))
head(data)
n<-length(names(data))
n
#划分数据集
set.seed(5)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.5, 0.5))
train <- data[ind == 1, ]
test <- data[ind == 2, ]
#寻找最优参数
grid<-expand.grid(nrounds=c(75,100),colsample_bytree=1,min_child_weight=1,eta=c(0.01,0.1,0.3),gamma=c(0.5,0.25),subsample=0.5,max_depth=c(2,3))
cntrl<-trainControl(method="cv",number=5,verboseIter = TRUE,returnData=FALSE,returnResamp="final")
set.seed(1)
train.xgb<-train(x=train[,1:n-1],y=train[,n],trControl=cntrl,tuneGrid=grid,method="xgbTree")
train.xgb
#设置xgboost数据矩阵
param<-list(objective="binary:logistic",booster="gbtree",eval_metric="error",eta=0.1,max_depth=3,subsample=0.5,colsample_bytree=1,gamma=0.25)
x<-as.matrix(train[,1:n-1])
x
y<-ifelse(train$class=="case",1,0)
y
train.mat<-xgb.DMatrix(data=x,label=y)
a<-as.matrix(test[,1:n-1])
a
b<-ifelse(test$class=="case",1,0)
b
test.mat<-xgb.DMatrix(data = a,label=b)
#创建模型
set.seed(10)
xgb.fit<-xgb.train(params = param,data=train.mat,nrounds = 75)
xgb.fit
#检查变量重要性（gain、cover、frequency）
impMatrix<-xgb.importance(feature_names = dimnames(x)[[2]],model = xgb.fit)
impMatrix
write.csv(impMatrix,"case-gain.csv")
xgb.plot.importance(impMatrix,main="Gain by Feature",col="lightblue")
pdf(file='xgboost-1-gain.pdf', height=6,width=6, family='GB1')
xgb.plot.importance(impMatrix,main="Gain by Feature",col="lightblue")
dev.off()
#查看模型在测试集上的表现
pred<-predict(xgb.fit,newdata = test.mat,type="response")
pred
#ROC
library(pROC)
roc.rf <- roc(response=b,predictor=as.numeric(pred))
roc.rf
plot(roc.rf, print.auc=TRUE,auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("lightblue", "lightblue"),max.auc.polygon=TRUE,auc.polygon.col="lightblue",max.auc.polygon.col="white",print.thres=TRUE)
pdf(file='xgboost-2-roc.pdf', height=6,width=6, family='GB1')
plot(roc.rf, print.auc=TRUE,auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("lightblue", "lightblue"),max.auc.polygon=TRUE,auc.polygon.col="lightblue",max.auc.polygon.col="white",print.thres=TRUE)
dev.off()
best<-coords(roc.rf, "best")
print(best)
#最优阈值下的混淆矩阵
pred<-ifelse(pred>best$threshold,1,0)
xgb.cf<-caret::confusionMatrix(as.factor(pred),as.factor(b),dnn = c('Actual', 'Predicted'),positive="1")
xgb.cf
b
b<-ifelse(b==0,"control","case")
pred
pred<-ifelse(pred==0,"control","case")
ggplot() + geom_confmat(aes(x =b, y = pred),normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction")+
  scale_fill_gradient2(low="white", high="lightblue")+theme_minimal()+theme_bw()
pdf(file='xgboost-3-hx.pdf', height=6,width=6, family='GB1')
ggplot() + geom_confmat(aes(x =b, y = pred),normalize = TRUE, text.perc = TRUE)+
  labs(x = "Reference",y = "Prediction")+
  scale_fill_gradient2(low="white", high="lightblue")+theme_minimal()+theme_bw()
dev.off()








