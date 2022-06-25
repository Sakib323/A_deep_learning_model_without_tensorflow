import math
import numpy as np
def sigmoid_numpy(x):
    sig=1/(1+math.exp(-x))
    return sig
def log_loss(y_true,y_predicted):
    epsilon=1e-15
    y_predicted_new=[max(i,epsilon)for i in y_predicted]
    y_predicted_new=[min(i,1-epsilon)for i in y_predicted_new]
    y_predicted_new=np.array(y_predicted_new)
    return -np.mean(y_true*math.log(y_predicted_new)+(1-y_true)*math.log(1-y_predicted_new))
class myNN():
    def __init__(self):
        self.w1=1
        self.w2=1
        self.bias=0
    def fit(self,X,y,epochs,loss_thresold):
        self.w1,self.w2,self.bias=self.gradient_descent(self,X['age'],X['affordibility'],y,epochs,loss_thresold)
    def predict(self,X_test):
        weighted_sum=self.w1*X_test['age']+self.w2*X_test['affordibility']+self.bias
        return sigmoid_numpy(weighted_sum)
def gradient_descent(self,age,affordibility,y_true,epochs,loss_thresold):
    w1,w2=1
    bias=0
    rate=0.5
    n=len(age)
    for i in range(epochs):
        weighted_sum=w1*age+w2*affordibility+bias
        y_predicted=sigmoid_numpy(weighted_sum)
        loss=log_loss(y_true,y_predicted)
        w1d=(1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
        w2d=(1/n)*np.dot(np.transpose(affordibility),(y_predicted-y_true))
        bias_d=np.mean(y_predicted-y_true)
        w1=w1-rate*w1d
        w2=w2-rate*w2d
        bias=bias-rate*bias_d
        if i%50==0:
            print(f'Epoch:{i},w1:{w1},w2:{w2},bias:{bias},loss:{loss}')
        if(loss<=loss_thresold):
            break
    return w1,w2,bias   