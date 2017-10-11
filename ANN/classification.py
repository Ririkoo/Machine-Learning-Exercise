import matplotlib.pyplot as plt
import numpy as np

d=[]
x1=[]
x2=[]
data_num=0
iter_time=100
plot_data_history=np.zeros((4,iter_time)) # 0:error_rate 1:w0(i) 2:w1(i) 3:w2(i)

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
		error_rate(Y,i)
		plot_data_history[1][i]=w0
		plot_data_history[2][i]=w1
		plot_data_history[3][i]=w2
	return w0,w1,w2

def draw_history_graph():
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

def hidden_layer(Z,W,w0):
	output_value=0.0
	fixed_value=10e-5  #(if needed)fix the accuracy of floating number 
	input_layer=np.dot(Z, W) + w0
	output_value=np.tanh(input_layer)
	return output_value

if __name__ == '__main__':
	data_num=read_data()
	w0,w1,w2=logistic_regression(-4,-2,4,0.05)

	# draw main graph(using NN to predict the value)
	x1_min,x1_max=np.array(x1).min()-.5, np.array(x1).max()+.5
	x2_min,x2_max=np.array(x2).min()-.5, np.array(x2).max()+.5
	x_axis, y_axis = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
	Z = hidden_layer(np.c_[x_axis.ravel(), y_axis.ravel()],np.array([w1,w2]),w0)
	Z = Z.reshape(x_axis.shape)
	plt.contourf(x_axis, y_axis, Z, alpha=0.8,cmap=plt.cm.Spectral)
	plot1=plt.scatter(x1, x2, s=100, c=d ,alpha=0.6)
	plot2=plt.scatter(5, 2.2 ,s=100 , c='green' ,marker='.')
	plt.legend([plot1, plot2], ('dataset','Test Point'), loc=2 ,numpoints=1)
	plt.xlim(x_axis.min(), x_axis.max())
	plt.ylim(y_axis.min(), y_axis.max())
	plt.title('Plot of x2 vs. x1')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()

	draw_history_graph()
	print "#Decision Boundary: w0,w1,w2 =({0},{1},{2})".format(w0,w1,w2)

