import pylab as plt
import numpy as np

plot_data_history=np.zeros((3,2000)) #0:MSE 1:a(i) 2:b(i)

def read_data(x,d):
	fopen = open('data.csv')
	for row in fopen:
		row_x,row_d=row.split(',')
		x.append(float(row_x))
		d.append(float(row_d))

def compute_mse(d,y,iter):
	mse=0.0
	for i in range(0,len(x)):
		mse+=(d[i]-y[i])**2
	mse=mse/2
	plot_data_history[0][iter]=mse
	return mse

def compute_fix_value(type,d,y):
	error=0
	for i in range(0,len(x)):
		if(type=='a'):
			error+=(-1*(d[i]-y[i])*x[i])
		if(type=='b'):
			error+=(-1*(d[i]-y[i]))
	return error

def update_y(a,b):
	y=[None]*len(x)
	for i in range(0,len(x)):
		y[i]=a*x[i]+b
	return y

def linear_regression(a,b,learning_rate,d,itertimes):
	y=[]
	for i in range(0,itertimes):
		y=update_y(a,b)
		compute_mse(d,y,i)
		plot_data_history[1][i]=a
		plot_data_history[2][i]=b
		a=a-learning_rate*compute_fix_value('a',d,y)
		b=b-learning_rate*compute_fix_value('b',d,y)
	return a,b

def draw_graph():
	# Plot y v.s. x
	plot1,=plt.plot(x,np.array(x)*a+b)
	plot2,=plt.plot(x, d ,'or')
	plt.title('Plot of y vs. x')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend([plot1, plot2], ('regression result', 'dataset'), 'best', numpoints=1)
	plt.xlim(0.0, 10.0)
	plt.ylim(0.0, 30.0)
	plt.show()

	# #Plot of MSE vs. iteration times
	tmpx=np.linspace(0, 2000, 2000)
	plot1,=plt.plot(tmpx, np.zeros(2000))
	plot2,=plt.plot(tmpx, plot_data_history[0],'or')
	plt.title('Plot of MSE vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('MSE')
	plt.xlim(0.0, 20.0)
	plt.ylim(-2.0, 10.0)
	plt.show()

	# Plot of a vs. iteration times
	tmpx=np.linspace(0, 2000, 2000)
	plot2,=plt.plot(tmpx, plot_data_history[1],'or')
	plt.title('Plot of a vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('a')
	plt.xlim(0.0, 30.0)
	plt.ylim(0.0, 3.0)
	plt.show()
	
	#Plot of b vs. iteration times
	tmpx=np.linspace(0, 2000, 2000)
	plot2,=plt.plot(tmpx, plot_data_history[2],'or')
	plt.title('Plot of b vs. iteration times')
	plt.xlabel('iteration times')
	plt.ylabel('b')
	plt.xlim(0.0, 30.0)
	plt.ylim(0.0, 3.0)
	plt.show()


if __name__ == '__main__':
	x=[]
	d=[]
	read_data(x,d)
	a,b=linear_regression(1.5,0.8,0.02,d,2000)
	print "#Linear Regression Result:a={0},b={1}".format(a,b)
	draw_graph()

