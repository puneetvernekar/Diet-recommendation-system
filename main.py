from tkinter import *
from tkinter import simpledialog
from sklearn.cluster import KMeans
import tkinter as tk
import pandas as pd
import numpy as np
main_win = Tk()


def Healthy():
    print(" Age: %s\n Weight: %s\n Height: %s\n" % (e1.get(), e3.get(), e4.get()))
    
    ROOT = tk.Tk()
    
    ROOT.withdraw()

    data=pd.read_csv('dataset.csv')       ##Reading the csv file
    data.head(5)

    fooddata=data['food']               ##Stores the binary values of food items
    fooddataNumpy=fooddata.to_numpy()   ##Converts the normal list to an numpy array

    fooditems=data['fooditems']         ##Stores the names of the food items
    
    foodid=[]
    for i in range(len(fooddata)):
      if fooddataNumpy[i]==1:
            foodid.append(i)            ##Stores the respective value of food items
    
    
    foodiddata = data.iloc[foodid]        ##Stores only those food items which have 1 in their binary value
    foodiddata=foodiddata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    foodiddata=foodiddata.iloc[Valapnd]
    foodiddata=foodiddata.T

    age=int(e1.get())
    weight=float(e3.get())
    height=float(e4.get())
    bmi = weight/(height**2) 
    agewiseinp=0

    for lp in range (0,100,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)

    print("Your body mass index is: ", bmi)
    if ( bmi < 16):
        print("severely underweight")
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi=1
    elif ( bmi >=30):
        print("severely overweight")
        clbmi=0   
    
    foodiddata=foodiddata.to_numpy()
    ti=(bmi+agecl)/2

    ###################################################################################################################DOUBT---------------
    
    #KMEANS
    Datacalorie=foodiddata[1:,1:len(foodiddata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=5, random_state=10).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    
    inp=[]
    datafin=pd.read_csv('data.csv')
    datafin.head(5)
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    healthycatDdata=healthycat.to_numpy()
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float64)
    t=0
    r=0
    s=0
    yt=[]

    for zz in range(5):
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1


    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float64)
    for jj in range(len(healthycat)):
        valloc=list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    #####################################################################################################----------------


    #####################################################################################################
    #######SKLEARN--------------(TRAINING)

    from sklearn.model_selection import train_test_split
    
    X_train= healthycatfin
    y_train=yt
    
    from sklearn.model_selection import train_test_split

    from sklearn.ensemble import RandomForestClassifier
    
    clf=RandomForestClassifier(n_estimators=100)
    
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    
    print ("SUGGESTED FOOD ITEMS -->>")
    for ii in range(len(y_pred)):
        if y_pred[ii]==2:
            print(fooditems[ii])


Label(main_win,text="Age :",font='Helvetica 12 ').grid(row=1,column=0,sticky=W,pady=4)
Label(main_win,text="Weight (Kg) :",font='Helvetica 12 ').grid(row=2,column=0,sticky=W,pady=4)
Label(main_win,text="Height (Metre) :", font='Helvetica 12 ').grid(row=3,column=0,sticky=W,pady=4)

e1 = Entry(main_win,bg="light grey")
e3 = Entry(main_win,bg="light grey")
e4 = Entry(main_win,bg="light grey")
e1.focus_force() 

e1.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)

Button(main_win,text='Quit',font='Helvetica 8 bold',command=main_win.quit).grid(row=5,column=0,sticky=W,pady=4)
Button(main_win,text='Recommend',font='Helvetica 8 bold',command=Healthy).grid(row=4,column=0,sticky=W,pady=4)
main_win.geometry("400x200")
main_win.wm_title("DIET RECOMMENDATION SYSTEM")
main_win.mainloop()
