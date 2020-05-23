import sys
import os

fi=open("accuracy","r")
xx=fi.readline()
acc=float(xx)
f=open("input.txt","r")
x=[]
file=f.readlines()
for line in file:
    x.append(int(line.strip()))
a=open("accuracy.old","r")
b=a.readline()
acc_old=float(b)
a.close()
fi.close()
f.close()

old_kernel=x[0]
old_layer=x[1]
'''
kernel=x[0]        3
x[1]=x[1]         1
pool=x[2]          2
filter_size=x[3]   32

'''

if acc !=acc_old:
       

    if x[0]==5:

        if x[1]==2:
          x[1]+=1  #x[1] increased
          os.system("cp input.txt input.old")
          os.system("mv accuracy  accuracy.old") 
        elif x[1]==3 and x[0]==5:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
            x[0]+=2
        elif x[1]==4 and x[0]==5:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
            x[0]+=2    
        elif x[1]==5 and x[0]==5:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv  input.old input.txt")
            x[0]+=2   
             
            
        elif x[1]==6 and x[0]==5:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
            x[0]+=2    
        elif x[1]==7 and x[0]==5:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.old input.txt")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
            x[0]+=2    
    elif x[0]==7:
        if x[1]==2:
            x[1]+=1  #x[1] increased
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old") 
        elif x[1]==3:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
            
        elif x[1]==4:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
                
        elif x[1]==5:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
                
            
        elif x[1]==6:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
                
        elif x[1]==7:
          if acc>acc_old and (acc-acc_old)>0.0001:
            x[1]+=1  # if acc is increasing and kernal size is still 5 then increase 1 more x[1]
            os.system("cp input.txt input.old")
            os.system("mv accuracy  accuracy.old")
          else:
            os.system("mv input.old input.txt")
          
elif acc==acc_old:
    print("call job 6 its the best i can do    best acc: ",acc) 
    exit();                        
else:
    print("something wrong") 
   
f=open("input.txt","r")
y=[]
file=f.readlines()
for line in file:
    y.append(int(line.strip()))
f.close()

   
if x[0] > old_kernel: 
    f=open("input.txt","w")
    f.write(str(x[0])+"\n")        
    f.write(str(y[1])+"\n")        
    f.write(str(x[2])+"\n")               
    f.close()
    print("file updated sucessfully")
elif x[1] >old_layer:   
    f=open("input.txt","w")
    f.write(str(y[0])+"\n")        
    f.write(str(x[1])+"\n")        
    f.write(str(x[2])+"\n")             
    f.close()
else:
    print("using old input ")   
