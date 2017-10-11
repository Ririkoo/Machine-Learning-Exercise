import matplotlib.pyplot as plt
import numpy as np

d=[]
x1=[]
x2=[]
data_num=0
iter_time=30
plot_data_history=np.zeros((4,iter_time)) #0:error_rate 1:w0(i) 2:w1(i) 3:w2(i)

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

def error_rate(y,cur_iter):
	F=0
	E=0.0
	for i in range(0,data_num):
			E+=(d[i]-y[i])**2
	E=E/(data_num)
	plot_data_history[0][cur_iter]=E
	return E


def logistic_regression(w0,w1,w2,learning_rate):
	for i in range(0,iter_time):
		Y=[]
		for j in range(0,data_num):
			hypothesis=w0+w1*x1[j]+w2*x2[j]
			y=np.tanh(hypothesis)
			p_hypo=1-(np.tanh(hypothesis))**2
			w0+=learning_rate*(d[j]-y)*p_hypo
			w1+=learning_rate*(d[j]-y)*x1[j]*p_hypo
			w2+=learning_rate*(d[j]-y)*x2[j]*p_hypo
			Y.append(y)
		print error_rate(Y,i)
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
	# plot3=plt.scatter(5, 2.2 ,s=100 , c='g')
	#plt.legend([plot1, plot2 ,plot3], ('dataset', 'classification' ,'Test Point'), loc=2 ,numpoints=1)
	plt.title('Plot of x2 vs. x1')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()
	# Plot of error_rate vs. iteration times
	tmpx=np.linspace(0, iter_time, iter_time)
	plt.scatter(tmpx, plot_data_history[0] ,s=40)
	plt.title('Plot of error rate vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('error rate')
	plt.show()
	# Plot of w0 vs. iteration times
	tmpx=np.linspace(0, iter_time, iter_time)
	plt.scatter(tmpx, plot_data_history[1] ,s=40)
	plt.title('Plot of w0 vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('w0')
	plt.show()
	# Plot of w1 vs. iteration times
	tmpx=np.linspace(0, iter_time, iter_time)
	plt.scatter(tmpx, plot_data_history[2] ,s=40)
	plt.title('Plot of w1 vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('w1')
	plt.show()
	# Plot of w3 vs. iteration times
	tmpx=np.linspace(0, iter_time, iter_time)
	plt.scatter(tmpx, plot_data_history[3] ,s=40)
	plt.title('Plot of w2 vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('w2')
	plt.show()

if __name__ == '__main__':
	data_num=read_data()
	w0,w1,w2=logistic_regression(5,-4,5,0.2)
	draw_graph()
	print "#Decision Boundary: {0} + x1*{1} + x2*{2} = 0".format(w0,w1,w2)

