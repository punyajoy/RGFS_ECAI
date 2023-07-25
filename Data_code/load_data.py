import pandas as pd
import os 

def load_data_own_new_adversarial(data_path='Davidson',number_of_samples=50, random_seed=2021):
    path_own=os.path.dirname(os.path.realpath(__file__))[:-9]
    temp_path_train=path_own+'Dataset/NEW/Adversarial_Attacks/' + data_path + '_50_2021.csv'
    df_train=pd.read_csv(temp_path_train)

    temp_path_val=path_own+'Dataset/NEW/'+data_path+'/'+'Val.csv'
    print("Loading Validation Data from " + temp_path_val + "...")
    df_val=pd.read_csv(temp_path_val)
    temp_path_test=path_own+'Dataset/NEW/'+data_path+'/'+'Test.csv'
    print("Loading Test Data from " + temp_path_test + "...")
    df_test=pd.read_csv(temp_path_test)
    
    temp_path_total=path_own+'Dataset/NEW/'+data_path+'/'+'Total_train.csv'
    df_rest=pd.read_csv(temp_path_total)
    list_temp=list(df_train[df_train.columns[0]])
    df_rest=df_rest[~df_rest[df_rest.columns[0]].isin(list_temp)]
    return df_train,df_val,df_test,df_rest


def load_data_own_new(data_path='Davidson',number_of_samples=50, random_seed=2021):
    path_own=os.path.dirname(os.path.realpath(__file__))[:-9]
    if(number_of_samples==-1):
        temp_path_train=path_own+'Dataset/NEW/'+data_path+'/'+'Total_train.csv'
        df_train=pd.read_csv(temp_path_train)
    else:
        temp_path_train=path_own+'Dataset/NEW/'+data_path+'/'+str(number_of_samples)+'/'+'Train_'+str(random_seed)+'.csv'
        print("Loading Training Data from " + temp_path_train + "...")
        df_train=pd.read_csv(temp_path_train)
        
    
    
    temp_path_val=path_own+'Dataset/NEW/'+data_path+'/'+'Val.csv'
    print("Loading Validation Data from " + temp_path_val + "...")
    df_val=pd.read_csv(temp_path_val)
    temp_path_test=path_own+'Dataset/NEW/'+data_path+'/'+'Test.csv'
    print("Loading Test Data from " + temp_path_test + "...")
    df_test=pd.read_csv(temp_path_test)
    
    temp_path_total=path_own+'Dataset/NEW/'+data_path+'/'+'Total_train.csv'
    df_rest=pd.read_csv(temp_path_total)
    list_temp=list(df_train[df_train.columns[0]])
    df_rest=df_rest[~df_rest[df_rest.columns[0]].isin(list_temp)]
    return df_train,df_val,df_test,df_rest


def load_data_own(data_path='Davidson',number_of_samples=50, random_seed=2020):
    path_own=os.path.dirname(os.path.realpath(__file__))[:-9]
    if(number_of_samples==-1):
        temp_path_train=path_own+'Dataset/'+data_path+'/'+'Total_Train.csv'
        df_train=pd.read_csv(temp_path_train)
    else:
        temp_path_train=path_own+'Dataset/'+data_path+'/'+str(number_of_samples)+'/'+'Train_'+str(random_seed)+'.csv'
        df_train=pd.read_csv(temp_path_train)
    
    
    temp_path_val=path_own+'Dataset/'+data_path+'/'+'Val.csv'
    df_val=pd.read_csv(temp_path_val)
    temp_path_test=path_own+'Dataset/'+data_path+'/'+'Test.csv'
    df_test=pd.read_csv(temp_path_test)
    
    temp_path_total=path_own+'Dataset/'+data_path+'/'+'Total_train.csv'
    df_rest=pd.read_csv(temp_path_total)
    list_temp=list(df_train[df_train.columns[0]])
    df_rest=df_rest[~df_rest[df_rest.columns[0]].isin(list_temp)]
    return df_train,df_val,df_test,df_rest


def load_data_own_swr(data_path='Davidson',number_of_samples=50, random_seed=2020):
    path_own=os.path.dirname(os.path.realpath(__file__))[:-9]
    if(number_of_samples==-1):
        temp_path_train=path_own+'Dataset/'+data_path+'/'+'Total_Train.csv'
        df_train=pd.read_csv(temp_path_train)
    else:
        temp_path_train=path_own+'Dataset/'+data_path+'/'+str(number_of_samples)+'/'+'Train_'+str(random_seed)+'.csv'
        df_train=pd.read_csv(temp_path_train)
    
    
    temp_path_val=path_own+'Dataset/'+data_path+'/'+'Val.csv'
    df_val=pd.read_csv(temp_path_val)
    temp_path_test=path_own+'Dataset/'+data_path+'/'+'Test_swr.csv'
    df_test=pd.read_csv(temp_path_test)
    
    temp_path_total=path_own+'Dataset/'+data_path+'/'+'Total_train.csv'
    df_rest=pd.read_csv(temp_path_total)
    list_temp=list(df_train[df_train.columns[0]])
    df_rest=df_rest[~df_rest[df_rest.columns[0]].isin(list_temp)]
    return df_train,df_val,df_test,df_rest



