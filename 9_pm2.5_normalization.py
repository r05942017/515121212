from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import winsound
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

data = pd.read_csv('train.csv',encoding='big5', usecols=lambda x: x not in ['日期','測站','測項'])

data=DataFrame(data)
#test=DataFrame(data[:18])
print(len(data))

datalenth=len(data)/18
AM_TEMP=data[0:18]

AM_TEMP=AM_TEMP.rename(index={0:'AMB_TEMP',1:'CH4',2:'CO',3:'NMHC',4:'NO',5:'NO2',6:'NOx',7:'O3',8:'PM10',9:'PM2.5',
						10:'RAINFALL',11:'RH',12:'SO2',13:'THC',14:'WD_HR',15:'WIND_DIREC',16:'WIND_SPEED',17:'WS_HR'})

for i in range(1,int(len(data)/18),1):      #把相同item排在同一列
	j=18*i
	buf=data[j:j+18]
	buf=buf.rename(index={j:'AMB_TEMP',j+1:'CH4',j+2:'CO',j+3:'NMHC',j+4:'NO',j+5:'NO2',j+6:'NOx',j+7:'O3',j+8:'PM10',j+9:'PM2.5',
						j+10:'RAINFALL',j+11:'RH',j+12:'SO2',j+13:'THC',j+14:'WD_HR',j+15:'WIND_DIREC',j+16:'WIND_SPEED',j+17:'WS_HR'})
	AM_TEMP = pd.concat([AM_TEMP,buf],axis=1,ignore_index=True)


AM_TEMP=AM_TEMP.T   #矩陣旋轉

for i in range(0,int(len(AM_TEMP)),1):   #change NR to 0
	if AM_TEMP['RAINFALL'][i]=='NR':
		AM_TEMP['RAINFALL'][i]=0

#print(AM_TEMP['PM2.5'][5500])

#----------------------------------------------------------------------------------------------
#print(len(AM_TEMP))
'''
rmse= 6.367920855592217
b= 0.823969860181 w= [-0.12725916  0.08743143  0.29104508 -0.34143819  0.00969034  0.47631491
 -0.71780158  0.3937045   0.8749948 ]
'''

num=9
b=0
w=np.zeros(num)


lr=1	#learning rate
iteration=250
lr_b=0
lr_w=np.zeros(num)
lambd=0

#print('rando=',np.random.random_sample((5,)))
#b=4*np.random.random()
#w=4*np.random.random_sample((num,))
print(w)
#y=b+wx
AM_TEMP=AM_TEMP.apply(pd.to_numeric)    #str to float

y_correct=DataFrame()
for cor in range(12):				#get y_correct
	if cor==0:
		y_correct=AM_TEMP['PM2.5'][num:480-120]
	else:
		y_correct=y_correct.append( AM_TEMP['PM2.5'][cor*480+num:(cor+1)*480-120], ignore_index=True)
#	print('ylen=',len(y_correct),'cor=',cor)
print(y_correct)
run=1
val=AM_TEMP.values
val2=val
AM_TEMP=AM_TEMP.values

pm25mean=np.mean(AM_TEMP[:,9])#---------------------------------------------------------
pm25std=np.std(AM_TEMP[:,9])
print('mean=',pm25mean,'std=',pm25std)

for i in range(18):
	AM_TEMP[:,i]=(AM_TEMP[:,i]-np.mean(AM_TEMP[:,i]))/np.std(AM_TEMP[:,i])#------------------------

#while run==1:
#b=5*np.random.random()
#w=5*np.random.random_sample((num,))
for i in range(iteration):
	#num=1
	b_grade=0.0
	w_grade=np.zeros(num)

	j=0
	print(i)
	while run==1:
		#print(AM_TEMP[j+num,9])
		equ=2.0*(AM_TEMP[j+num,9]-b-np.dot(w,AM_TEMP[j:j+num,9]))

		#print('num=',num)
		b_grade=b_grade-equ*1.0

		if num==1:
			w_grade[0]=w_grade[0]-equ*AM_TEMP[j,9]
		else:
			for k in range(num):
				w_grade[k]=w_grade[k]-equ*AM_TEMP[j+k,9]
		if (j+num+1)%360==0:
			j=j+num+120
		
		j=j+1
		if j>=(5760-num):
			break
	#	num=num+1
	for j in range(num):						#regulation
		w_grade[j]=w_grade[j]+2*lambd*w[j]
	lr_b=lr_b+b_grade**2
	b=b-lr/np.sqrt(lr_b)*b_grade

	for a in range(num):
		lr_w[a]=lr_w[a]+w_grade[a]**2
		w[a]=w[a]-lr/np.sqrt(lr_w[a])*w_grade[a]

	y_pred=np.zeros(5760-12*num)
	for cor in range(12):
		buf=b

		for x in range(num):
			buf=buf+AM_TEMP[cor*480+x:((cor+1)*480-num+x)-120,9]*w[x]
		if cor==0:
			y_pred=buf
		else:
			y_pred=np.hstack((y_pred,buf))
	#y_pred=y_pred+lambd*np.sum(np.square(w))				#regulation
	y_pred=y_pred*pm25std+pm25mean							#renormaztion

	print('corrlen=',len(y_correct),'predlen=',len(y_pred))
	pred_rmse=rmse(y_pred,y_correct.values)
	print('rmse=',float(pred_rmse))
	print('b=',b,'w=',w)
	print('len',len(y_pred))

#print('y=\n',y_pred)
#print('Y_pred',y_pred)



#-----------------------------------------------
'''
y_validation=DataFrame()
for cor in range(12):				#get y_validation
	for x in range(5):
		if cor==0 and x==0:
			y_validation=val[360+num:360+24,9]
		else:
			y_validation=np.hstack((y_validation,val[360+cor*480+num+24*x:360+cor*480+24*(x+1),9]))

#print('y_val=',len(y_validation))
for cor in range(12):
	buf=b
	for y in range(5):
		for x in range(num):
			buf=buf+AM_TEMP[360+cor*480+x+y*24:360+cor*480+x+(1+y)*24-num,9]*w[x]
		if cor== 0 and y==0:
			y_test=buf
		else:
			y_test=np.hstack((y_test,buf))


#print('y_test=',len(y_test))
vali_rmse=rmse(y_test,y_validation)
print('val_rmse=',vali_rmse)
'''
#-------------------------------------------------------------------------------------------------val2

y_validation=DataFrame()
for cor in range(12):				#get y_validation2
	if cor==0:
		y_validation=val[436+num:480,9]
	else:
		y_validation=np.hstack((y_validation,val[436+cor*480+num:480+cor*480,9]))

for i in range(18):
	val[:,i]=(val[:,i]-np.mean(val[:,i]))/np.std(val[:,i])#------------------------ normalization
#print('y_val=',len(y_validation))
#y_test=np.zeros(1440-12*num)
for cor in range(12):
	buf=b
	for x in range(num):
		buf=buf+val[cor*480+x+436:cor*480+x+480-num,9]*w[x]
	if cor==0:
		y_test=buf
	else:
		y_test=np.hstack((y_test,buf))
y_test=y_test+lambd*np.sum(np.square(w))
y_test=y_test*pm25std+pm25mean							#renormaztion
#print('y_test2=',len(y_test))
vali_rmse2=rmse(y_test,y_validation)
print('val_rmse2=',vali_rmse2)
	#if pred_rmse-vali_rmse2>0 and pred_rmse<7.15 :
	#	print('success')
	#	break



winsound.Beep(600,1000)