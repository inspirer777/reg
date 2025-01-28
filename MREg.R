library(alr4)
library(readxl)
library(UsingR)
library(ggplot2)
library(nortest)
library(leaps)
library(car)
#basic statistic informations#
data<- read_excel("C:/Users/BEHINLAPTOP/Desktop/regression/D.csv")
View(data)
attach(data)
names(data)
summary(data)
head(data)
tail(data)
cv <- cor(data)
round(cv,3)
#Figure 6.21 on page 178
pairs(y~x1+x2+x3+x4+x5+x6+x7)

scatterplot(x=x1, y=y)
scatterplot(x=x2, y=y)
scatterplot(x=x3, y=y)
scatterplot(x=x4, y=y)
scatterplot(x=x5, y=y)
scatterplot(x=x6, y=y)
scatterplot(x=x7, y=y)
###################################
lm<-lm(y~x1+x2+x3+x4+x5+x6+x7)
lm
summary(lm)
anova(lm)
yhat <- fitted(lm);yhat
confint(lm,level = 0.95)

lm1<-lm(y~x1)
#################################################
cor.test(x1,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

cor.test(x2,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

cor.test(x3,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

cor.test(x4,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

cor.test(x5,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

cor.test(x6,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

cor.test(x7,y,alternative = "two.sided",method = "pearson",conf.level = 0.98)

.
######################################
r1 <- resid(lm(y~x1))
qqnorm(r1)
plot(yhat,r1)
lillie.test(r1)

r2 <- resid(lm(y~x2))
qqnorm(r2)
plot(yhat,r2)
lillie.test(r2)

r3 <- resid(lm(y~x3))
qqnorm(r3)
plot(yhat,r3)
lillie.test(r3)

r4 <- resid(lm(y~x4))
qqnorm(r4)
plot(yhat,r4)
lillie.test(r4)

r5 <- resid(lm(y~x5))
qqnorm(r5)
plot(yhat,r5)
lillie.test(r5)

r6 <- resid(lm(y~x6))
qqnorm(r6)
plot(yhat,r6)
lillie.test(r6)

r7 <- resid(lm(y~x7))
qqnorm(r7)
plot(yhat,r7)
lillie.test(r7)

###################
c1<- predict(lm(y~x1),int="c");c1

c2<- predict(lm(y~x2),int="c");c2

c3<- predict(lm(y~x3),int="c");c3

c4<- predict(lm(y~x4),int="c");c4

c5<- predict(lm(y~x5),int="c");c5

c6<- predict(lm(y~x6),int="c");c6

c7<- predict(lm(y~x7),int="c");c7
#####################################
#regression diagnostics:
lm.influence(lm)
plot(lm)
abline(v=4/768)
abline(h=-2)
abline(h=2)

###################
sr=rstandard(lm)
qqnorm(lm$res)
qqline(lm$res)
lillie.test(lm$res)
hist(sr);rug(sr)
##############################
#Figure 3.26 on page 85
par(mfrow=c(2,2))
plot(lm)
par(mfrow=c(1,2))
StanRes1 <- rstandard(lm)
absrtsr1 <- sqrt(abs(StanRes1))

plot(x1,StanRes1,ylab="Standardized Residuals")
plot(x1,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

plot(x2,StanRes1,ylab="Standardized Residuals")
plot(x2,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

plot(x3,StanRes1,ylab="Standardized Residuals")
plot(x3,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

plot(x4,StanRes1,ylab="Standardized Residuals")
plot(x4,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

plot(x5,StanRes1,ylab="Standardized Residuals")
plot(x5,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

plot(x6,StanRes1,ylab="Standardized Residuals")
plot(x6,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

plot(x7,StanRes1,ylab="Standardized Residuals")
plot(x7,absrtsr1,ylab="Square Root(|Standardized Residuals|)")

######################################################################
#Figure 3.27 on page 86
#kernel:
#y(heating load)
par(mfrow=c(2,2))
sj <- bw.SJ(y,lower = 0.05, upper = 100)
plot(density(y,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="heating load");rug(y)
boxplot(y,ylab="heating load");qqnorm(y, ylab = "y");qqline(y, lty = 2, col=2)

#feshordegi nesbi (x1)
par(mfrow=c(2,2))
sj <- bw.SJ(x1,lower = 0.05, upper = 100)
plot(density(x1,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x1");rug(x1)
boxplot(x1,ylab="x1");qqnorm(x1, ylab = "x1");qqline(x1, lty = 2, col=2)

#masahat sath(x2)
par(mfrow=c(2,2))
sj <- bw.SJ(x2,lower = 0.05, upper = 100)
plot(density(x2,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x2");rug(x2)
boxplot(x2,ylab="x2");qqnorm(x2, ylab = "x2");qqline(x2, lty = 2, col=2)

#fazaye divar(x3)
par(mfrow=c(2,2))
sj <- bw.SJ(x3,lower = 0.05, upper = 100)
plot(density(x3,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x3");rug(x3)
boxplot(x3,ylab="x3");qqnorm(x3, ylab = "x3");qqline(x3, lty = 2, col=2)

#masahat saghf(x4)
par(mfrow=c(2,2))
sj <- bw.SJ(x4,lower = 0.05, upper = 100)
plot(density(x4,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x4");rug(x4)
boxplot(x4,ylab="x4");qqnorm(x4, ylab = "x4");qqline(x4, lty = 2, col=2)

#ertefae koli(x5)
par(mfrow=c(2,2))
sj <- bw.SJ(x5,lower = 0.05, upper = 100)
plot(density(x5,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x5");rug(x5)
boxplot(x5,ylab="x5")
qqnorm(x5, ylab = "x5")
qqline(x5, lty = 2, col=2)

#jahat(x6)
par(mfrow=c(2,2))
sj <- bw.SJ(x6,lower = 0.05, upper = 100)
plot(density(x6,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x6");rug(x6)
boxplot(x6,ylab="x6");qqnorm(x6, ylab = "x6");qqline(x6, lty = 2, col=2)

#mantaghe shishei(x7)
par(mfrow=c(2,2))
sj <- bw.SJ(x7,lower =0.001, upper = 20)
plot(density(x7,bw=sj,kern="gaussian"),type="l",
     main="Gaussian kernel density estimate",xlab="x7");rug(x7)
boxplot(x7,ylab="x7");qqnorm(x7, ylab = "x7");qqline(x7, lty = 2, col=2)
###################################################################
#Figure 3.30 on page 92
#box-cox:roykard 1:
#summary(tranxy <- powerTransform((cbind(x1+x2+x3+x4+x5+x6+x7)~1)))
#testTransform(tranxy,  1)
summary(tranxy0<-powerTransform(cbind(x1,x2,x3,x4)~1))
testTransform(tranxy0,(c(-1.26,1,1,1)))
summary(tranxy1<-powerTransform(cbind(x5,x6)~1))
testTransform(tranxy1,(c(0,0.6711,1)))
#summary(powerTransform(x7~1))

bx1<-x1^(-1.2584)
bx2<-x2
bx3<-x3
bx4<-x4
bx5<-log(x5)
bx6<-x6^(0.6711)
bx7<-x7
lmm<-lm(y~bx1+bx2+bx3+bx4+bx5+bx6+bx7)
summary(lmm)
############################################################
#invResPlot
invResPlot(lmm,key=TRUE)
#invResPlot & box-cox:
ay <- y^(-0.01)
lmm_inverse<- lm(ay~bx1+bx2+bx3+bx4+bx5+bx6+bx7)
pairs(ay~bx1+bx2+bx3+bx4+bx5+bx6+bx7)
summary(lmm_inverse)
#########################################
#roykarde 2 box-cox:
summary(tranxy0<-powerTransform(cbind(y,x1,x2,x3,x5,x6)~1))

vy <- y^(-1.5)
vx1<-x1^(-2.53)
vx2<-x2^(2.75)
vx3<-x3^(-1)
vx4<-x4
vx5<-log(x5)
vx6<-sqrt(x6)
vx7<-x7
lm2 <-lm(vy~vx1+vx2+vx3+vx4+vx5+vx6+vx7)
summary(lm2)
pairs(by~vx1+vx2+vx3+vx4+vx5+vx6+vx7)
########################################
plot(vx1,vy,xlab=expression(vx1))
abline(lsfit(vx1,vy))


#addvarible:
#Figure 6.27 on page 183
par(mfrow=c(2,4))
avPlot(lm2,variable=vx1,ask=FALSE,identify.points=FALSE)
avPlot(lm2,variable=vx2,ask=FALSE,identify.points=FALSE)
avPlot(lm2,variable=vx3,ask=FALSE,identify.points=FALSE)
avPlot(lm2,variable=vx4,ask=FALSE,identify.points=FALSE)
avPlot(lm2,variable=vx5,ask=FALSE,identify.points=FALSE)
avPlot(lm2,variable=vx6,ask=FALSE,identify.points=FALSE)
avPlot(lm2,variable=vx7,ask=FALSE,identify.points=FALSE)
mmps(lm2)
##########################################################
lm <- lm(y~(x1)+(x2)+(x3)+(x4)+(x5)+(x6)+(x7))# bedoone tabdil
lmm <- lm(y~bx1+bx2+bx3+bx4+bx5+bx6+bx7)#box-cox roykard 1
lm1 <- lm(ay~bx1+bx2+bx3+bx4+bx5+bx6+bx7)#invResPlot & box-cox
lm2 <- lm(vy~vx1+vx2+vx3+vx4+vx5+vx6+vx7)# box-cox
summary(lm)
summary(lmm)
summary(lm1)
summary(lm2)
par(mfrow=c(2,2))
invResPlot(lm,key=TRUE)
invResPlot(lmm,key=TRUE)
invResPlot(lm1,key=TRUE)
invResPlot(lm2,key=TRUE)

# regsubs
X <- cbind(bx1,bx2,bx3,bx4,bx5,bx6,bx7)
b<-regsubsets(as.matrix(X),ay)
summary(b)
rs <- summary(b)
par(mfrow=c(1,2))
plot(1:7,rs$adjr2,xlab="Subset Size",ylab="Adjusted R-squared")
subsets(b,statistic=c("adjr2"))
########################################

#Calculate adjusted R-squared
rs$adjr2
max(rs$adjr2)
lmd1 <- lm(ay~bx5)
lmd2 <- lm(ay~bx5+vx7)
lmd3 <- lm(ay~bx3+vx5+bx7)
lmd4 <- lm(ay~bx2+bx3+bx5+bx7)
lmd5 <- lm(ay~bx1+bx2+bx3+bx5+bx7)
lmd6 <- lm(ay~bx1+bx2+bx3+bx5+bx6+bx7)

summary(lmd1)
summary(lmd2)
summary(lmd3)
summary(lmd4)
summary(lmd5)
summary(lmd6)
summary(lmd7)
#######AIC########
n = 768
npar1 <- length(lmd1$coefficients) +1
2*npar1*(npar1+1)/(n-npar1-1)
extractAIC(lmd1,k=2)+2*npar1*(npar1+1)/(n-npar1-1)

npar2 <- length(lmd2$coefficients) +1
2*npar2*(npar2+1)/(n-npar2-1)
extractAIC(lmd2,k=2)+2*npar2*(npar2+1)/(n-npar2-1)

npar3 <- length(lmd3$coefficients) +1
2*npar3*(npar3+1)/(n-npar3-1)
extractAIC(lmd3,k=2)+2*npar3*(npar3+1)/(n-npar3-1)

npar4 <- length(lmd4$coefficients) +1
2*npar4*(npar4+1)/(n-npar4-1)
extractAIC(lmd4,k=2)+2*npar4*(npar4+1)/(n-npar4-1)

npar5 <- length(lmd5$coefficients) +1
2*npar5*(npar5+1)/(n-npar5-1)
extractAIC(lmd5,k=2)+2*npar5*(npar5+1)/(n-npar5-1)

npar6 <- length(lmd6$coefficients) +1
2*npar6*(npar6+1)/(n-npar6-1)
extractAIC(lmd6,k=2)+2*npar6*(npar6+1)/(n-npar6-1)

npar7 <- length(lmd7$coefficients) +1
2*npar7*(npar7+1)/(n-npar7-1)
extractAIC(lmd7,k=2)+2*npar4*(npar7+1)/(n-npar7-1)
################################################
#Calculate AICc
extractAIC(lmd1,k=2)+2*npar1*(npar1+1)/(n-npar1-1)
extractAIC(lmd2,k=2)+2*npar2*(npar2+1)/(n-npar2-1)
extractAIC(lmd3,k=2)+2*npar3*(npar3+1)/(n-npar3-1)
extractAIC(lmd4,k=2)+2*npar4*(npar4+1)/(n-npar4-1)
aic5 = extractAIC(lmd5,k=2)+2*npar5*(npar5+1)/(n-npar5-1)
aic6 = extractAIC(lmd6,k=2)+2*npar6*(npar6+1)/(n-npar6-1)
aic5
aic6
################################################
#Calculate BIC
extractAIC(lmd1,k=log(n))
extractAIC(lmd2,k=log(n))
extractAIC(lmd3,k=log(n))
extractAIC(lmd4,k=log(n))
extractAIC(lmd5,k=log(n))
extractAIC(lmd6,k=log(n))
extractAIC(lmd7,k=log(n))

#############################
#backward:
backAIC <- step(lm2,direction="backward")
summary(backAIC)
backBIC <- step(lm2,direction="backward", k=log(n))
summary(backBIC)
#################################
#forward:
mint <- lm(vy~1)
forwardAIC <- step(mint,scope=list(lower=~1, 
                                   upper=~bx1+bx2+bx3+bx4+bx5+bx6+bx7),
                   direction="forward")
summary(forwardAIC)
forwardBIC <- step(mint,scope=list(lower=~1, 
                                   upper=~bx1+bx2+bx3+bx4+bx5+bx6+bx7),
                   direction="forward",k=log(n))
summary(forwardBIC)
#####################################################
#stepwise:
StepwiseReg <- step(mint,scope=list(lower=~1, 
                                    upper=~bx1+bx2+bx3+bx4+bx5+bx6+bx7),
                    direction="both")
summary(StepwiseReg)

