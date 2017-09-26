import matplotlib.pyplot as plt
import numpy as np

d=[]
x1=[]
x2=[]
data_num=0
iter_time=10
plot_data_history=np.zeros((4,10)) #0:error_rate 1:w0(i) 2:w1(i) 3:w2(i) 

def read_data():
	cnt=0
	fopen = open('iris.txt')
	for row in fopen:
		rx1,rx2,rd=row.split('\t')
		x1.append(float(rx1))
		x2.append(float(rx2))
		d.append(float(rd))
		cnt+=1
	return cnt

# def sigmoid(z):
#     return 1/(1 + np.exp(-z))
# def cal_cost(y):
# 	E=0.0
# 	for i in range(0,data_num):
# 		if(d[i]==1):
# 			E+=-1*np.log(sigmoid(y[i]))
# 		else:
# 			E+=-1*np.log(sigmoid(1-y[i]))
# 	E=E/100	
# 	return E

def error_rate(y,cur_iter):
	F=0
	E=0.0
	for i in range(0,data_num):
		if(d[i]!=y[i]):
			F+=1
	E=F/float(data_num)
	plot_data_history[0][cur_iter]=E
	return E

def logistic_regression(w0,w1,w2,learning_rate):
	for i in range(0,iter_time):
		Y=[]
		for j in range(0,data_num):
			hypothesis=w0+w1*x1[j]+w2*x2[j]
			y=np.sign(hypothesis)
			w0+=learning_rate*(d[j]-y)
			w1+=learning_rate*(d[j]-y)*x1[j]
			w2+=learning_rate*(d[j]-y)*x2[j]
			#Y.append(hypothesis)
			Y.append(y)
		error_rate(Y,i)
		plot_data_history[1][i]=w0
		plot_data_history[2][i]=w1
		plot_data_history[3][i]=w2
	return w0,w1,w2

def draw_graph():
	# Plot x2 v.s. x1
	plot1=plt.scatter(x1, x2, s=100, c=d ,alpha=0.5)
	x_axis=np.linspace(0, 8, 100)
	y_axis=-(w0+x_axis*w1)/w2
	plot2,=plt.plot(x_axis, y_axis)
	plt.xlim(3.0, 8.0)
	plt.ylim(0.0, 6.0)
	plot3=plt.scatter(5, 2.2 ,s=100 , c='g')
	#plt.legend([plot1, plot2 ,plot3], ('dataset', 'classification' ,'Test Point'), 'best', numpoints=1)
	plt.title('Plot of x2 vs. x1')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()
	# Plot of error_rate vs. iteration times
	tmpx=np.linspace(0, 10, 10)
	plt.scatter(tmpx, plot_data_history[0] ,s=50)
	plt.title('Plot of error_rate vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('error_rate')
	plt.show()
	# Plot of w0 vs. iteration times
	tmpx=np.linspace(0, 10, 10)
	plt.scatter(tmpx, plot_data_history[1] ,s=50)
	plt.title('Plot of w0 vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('w0')
	plt.ylim(10.4, 10.6)
	plt.show()
	# Plot of w1 vs. iteration times
	tmpx=np.linspace(0, 10, 10)
	plt.scatter(tmpx, plot_data_history[2] ,s=50)
	plt.title('Plot of w1 vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('w1')
	plt.show()
	# Plot of w3 vs. iteration times
	tmpx=np.linspace(0, 10, 10)
	plt.scatter(tmpx, plot_data_history[3] ,s=50)
	plt.title('Plot of w3 vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('w3')
	plt.show()
	
if __name__ == '__main__':
	data_num=read_data()
	w0,w1,w2=logistic_regression(10,-8,3,0.01)
	draw_graph()
	print "#Decision Boundary: {0} + x1*{1} + x2*{2} = 0".format(w0,w1,w2)

