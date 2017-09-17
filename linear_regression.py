import matplotlib.pyplot as plt

def read_data(x,d):
	fopen = open('data.csv')
	for row in fopen:
		row_x,row_d=row.split(',')
		x.append(float(row_x))
		d.append(float(row_d))

def compute_mse(d,y):
	mse=0.0
	for i in range(0,len(x)):
		mse+=(d[i]-y[i])**2
	mse=mse/2
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
		compute_mse(d,y)
		a=a-learning_rate*compute_fix_value('a',d,y)
		b=b-learning_rate*compute_fix_value('b',d,y)
	return a,b

if __name__ == '__main__':
	x=[]
	d=[]
	read_data(x,d)
	a,b=linear_regression(1.5,0.8,0.02,d,2000)
	print "a={0},b={1}".format(a,b)

