import numpy as np
from Data_code.data import *
from Data_code.load_data import *
from Model_code.models import *
from Model_code.utils import *
from Eval_code.eval_scripts import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
import neptune.new as neptune
import GPUtil
import numpy as np
from datasets import list_datasets, load_dataset
from apiconfig import *
import pandas as pd
from tqdm import tqdm
import argparse
import json
import time
## training_type can take normal, domain_adap, domain_adap_ssda
## dataset Hatexplain, Founta, Davidson, Basile, Waseem, Olid
## model_path: model path as given in the huggingface
## logging: 'neptune' or 'local' (for printing the updates here)
## model_type: 'c' for classifier, 'r' is for rational_classifier, 't' is for target_classifier

params={
 'dataset_original':'Davidson',
 'training_points':-1,
 'model_path':'bert-base-uncased',
 'training_type':'normal',
 'logging':'local',
 'data_seed':42,
 'cache_path':'../../Saved_models/',
 'learning_rate':5e-5,
 'device':'cuda',
 'num_classes':3,
 'epochs':10,
 'batch_size':64,
 'rationale_impact':100,
 'target_impact':10,
 'attn_lambda':1,
 'train_rationale':False,
 'train_targets':False,
 'train_att':False,
 'save_path':'Saved_Models/Best_Toxic_BERT/'   
}


def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.9, maxMemory = 0.7, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)


def return_target_dict(train_dataset,valid_dataset,test_dataset, threshold=20):
    label_dict={}
    for row in train_dataset:
        for targets in row['annotators']['target']:
            for target in targets:
                try:    
                    label_dict[target]+=1
                except KeyError:
                    label_dict[target]=0
    
    for row in valid_dataset:
        for targets in row['annotators']['target']:
            for target in targets:
                try:    
                    label_dict[target]+=1
                except KeyError:
                    label_dict[target]=0
    
    for row in test_dataset:
        for targets in row['annotators']['target']:
            for target in targets:
                try:    
                    label_dict[target]+=1
                except KeyError:
                    label_dict[target]=0
    
    
    print(label_dict)
    
    target_dict = {}
    j=0
    
    for target in label_dict.keys():
        if label_dict[target] >= 20:
            target_dict[target]=j
            j+=1
    #target_dict['Other']=j
     
    return target_dict

    

def train(training_dataloader, validation_dataloader, test_dataloader, model, tokenizer, params,run):
    epochs=params['epochs']
    total_steps = len(training_dataloader) * epochs
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['learning_rate'], eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    best_macro_f1_val = 0
    best_macro_f1_val_rat = 0
    best_macro_f1_test = 0
    best_accuracy_test = 0
    best_pre_test = 0
    best_rec_test = 0
    
    best_macro_f1_rat_test = 0
    best_acc_rat_test =0
    best_pre_rat_test = 0
    best_rec_rat_test = 0
    
    
    
    best_macro_f1_target_test = 0
    best_acc_target_test =0
    best_pre_target_test = 0
    best_rec_target_test = 0
    
    best_model = None
    # current_epoch, best_weighted_f1 = load_metrics(filepath, model, optimizer)

    criterion = nn.CrossEntropyLoss()

    for epoch_i in tqdm(range(0, epochs)):
        model.train()
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            if(params['dataset']=='HateXplain'):
                b_input_ids=batch[0].to(device) 
                b_input_mask=batch[1].to(device)
                b_attn = batch[2].to(device).long()
                b_targets = batch[3].to(device)
                b_labels = batch[4].to(device).long()
            else:
                b_input_ids=batch[0].to(device) 
                b_input_mask=batch[1].to(device)
                b_labels = batch[3].to(device).long()
            if(params['dataset']=='HateXplain'):
                if(params['train_rationale']==True and params['train_targets']==True):
                    ypred, _, logits, loss = model(b_input_ids, b_input_mask, attn=b_attn,targets=b_targets,labels=b_labels)
                elif(params['train_rationale']==True and params['train_targets']==False):
                    ypred, _, logits, loss = model(b_input_ids, b_input_mask, attn=b_attn,targets=None,labels=b_labels)           
                elif(params['train_att']==True):
                    ypred, loss = model(b_input_ids, b_input_mask, attn=b_attn,labels=b_labels)
                else:    
                    ypred, loss = model(b_input_ids, b_input_mask,labels=b_labels)
            else:
                 ypred, loss = model(b_input_ids, b_input_mask,labels=b_labels)
            
            #loss = loss + params['rationale_impact']*criterion(ypred,b_labels)
            # print(outputs.logits)
            if(params['logging']=='local'):
                if step%100 == 0:
                    print(loss.item())
            else:
                run["train/batch_loss"].log(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        
        print("running validation")
        macro_f1_val,accuracy_val, pre_val, rec_val = evaluate_classifier(validation_dataloader, params,model,device)
        if(params['train_rationale']==True):
            macro_f1_val_rat,accuracy_val_rat, pre_rat_val, rec_rat_val = evaluate_rationales(validation_dataloader,params, model,device)
        if(params['train_targets']==True):
            macro_f1_val_tar,accuracy_val_tar, pre_tar_val, rec_tar_val = evaluate_target_classifier(validation_dataloader, params,model,device)
        
        print("running test")
        macro_f1_test,accuracy_test, pre_test, rec_test = evaluate_classifier(test_dataloader, params,model,device)
        if(params['train_rationale']==True):
            macro_f1_test_rat,accuracy_test_rat, pre_rat_test, rec_rat_test = evaluate_rationales(test_dataloader, params,model,device)
        if(params['train_targets']==True):
            macro_f1_test_tar,accuracy_test_tar, pre_tar_test, rec_tar_test = evaluate_target_classifier(test_dataloader, params,model,device)
            
        
        if(params['logging']=='neptune'):
            #### val scores updated
            run["label/val/f1"].log(macro_f1_val)
            run["label/val/accuracy"].log(accuracy_val)
            run["label/val/positive_class_precision"].log(pre_val)
            run["label/val/positive_class_recall"].log(rec_val)
            
            #### test scores updated
            run["label/test/f1"].log(macro_f1_test)
            run["label/test/accuracy"].log(accuracy_test)
            run["label/test/positive_class_precision"].log(pre_test)
            run["label/test/positive_class_recall"].log(rec_test)
            
            if(params['train_rationale']==True):
                #### val rationale updated
                run["rationale/val/f1"].log(macro_f1_val_rat)
                run["rationale/val/accuracy"].log(accuracy_val_rat)
                run["rationale/val/positive_class_precision"].log(pre_rat_val)
                run["rationale/val/positive_class_recall"].log(rec_rat_val)

                #### test rational updated
                run["rationale/test/f1"].log(macro_f1_test_rat)
                run["rationale/test/accuracy"].log(accuracy_test_rat)
                run["rationale/test/positive_class_precision"].log(pre_rat_test)
                run["rationale/test/positive_class_recall"].log(rec_rat_test)
            
        else:
            print("  Macro F1 Val: {0:.3f}".format(macro_f1_val))
            print("  Macro F1 Test: {0:.3f}".format(macro_f1_test))
            
            if(params['train_rationale']==True):
                    print("  Macro F1 Rationale Val: {0:.3f}".format(macro_f1_val_rat))
                    print("  Macro F1 Rationale Test: {0:.3f}".format(macro_f1_test_rat))
            
            if(params['train_targets']==True):
                    print("  Macro F1 Target Val: {0:.3f}".format(macro_f1_val_tar))
                    print("  Macro F1 Target Test: {0:.3f}".format(macro_f1_test_tar))
            
            
            
        if (macro_f1_val > best_macro_f1_val):
            #best_macro_f1_val_rat=macro_f1_val_rat
            best_macro_f1_val = macro_f1_val
            best_macro_f1_test = macro_f1_test
            best_accuracy_test = accuracy_test
            best_pre_test = pre_test
            best_rec_test = rec_test
            if(params['train_rationale']==True):
                best_acc_rat_test =accuracy_test_rat
                best_macro_f1_rat_test =macro_f1_test_rat
                best_pre_rat_test = pre_rat_test
                best_rec_rat_test = rec_rat_test
            if(params['train_targets']==True):
                best_acc_tar_test =accuracy_test_tar
                best_macro_f1_tar_test =macro_f1_test_tar
                best_pre_tar_test = pre_tar_test
                best_rec_tar_test = rec_tar_test
            save_bert_model_all_but_one(model,tokenizer,params)

################### Only for normal datasets
            #save_bert_model(model.bert,tokenizer,params)



#             best_model = copy.deepcopy(model)
#             save_metrics(filepath, epoch_i, model, optimizer, weighted_f1)
    
    if(params['logging']!='local'):
        if(params['train_rationale']==True):
            run["rationale/test/best_f1"].log(best_macro_f1_rat_test)
            run["rationale/test/best_accuracy"].log(best_acc_rat_test)
            run["rationale/test/best_positive_class_precision"].log(best_pre_rat_test)
            run["rationale/test/best_positive_class_recall"].log(best_rec_rat_test)        

            
        run["label/test/best_f1"].log(best_macro_f1_test)
        run["label/test/best_accuracy"].log(best_accuracy_test)
        run["label/test/best_positive_class_precision"].log(best_pre_test)
        run["label/test/best_positive_class_recall"].log(best_rec_test)        

    else:
        print("  Macro F1 Test: {0:.3f}".format(best_macro_f1_test))
        print("  Accuracy Test: {0:.3f}".format(best_accuracy_test))

        if(params['train_rationale']==True):
            print("  Macro F1 Rationale Test: {0:.3f}".format(best_macro_f1_rat_test))
            print("  Accuracy Rationale Test: {0:.3f}".format(best_acc_rat_test))

        if(params['train_targets']==True):
            print("  Macro F1 Target Test: {0:.3f}".format(best_macro_f1_tar_test))
            print("  Accuracy Target Test: {0:.3f}".format(best_acc_tar_test))

            
            
            
            
            
            
            

def train_caller(params,dataset,run=None):
    params['dataset']=dataset
    if(params['training_type']=='normal'):
        tokenizer = AutoTokenizer.from_pretrained(params['model_path'], use_fast=False, cache_dir=params['cache_path'])
        ### add model loading code 
        if(params['dataset']=='HateXplain'):
            dataset = load_dataset('hatexplain', split = ['train', 'validation', 'test'])
            train_dataset = dataset[0]
            valid_dataset = dataset[1]
            test_dataset = dataset[2]
            target_dict=return_target_dict(train_dataset, valid_dataset, test_dataset)
            
            print(len(target_dict))
            params['targets_num']=len(target_dict)
            if(params['logging']=='neptune'):
                run["parameters"] = params
            train_data_source = Hatexplain_Dataset_Target(train_dataset, params, tokenizer,target_dict,train = True)
            val_data_source = Hatexplain_Dataset_Target(valid_dataset, params, tokenizer, target_dict)
            test_data_source = Hatexplain_Dataset_Target(test_dataset, params, tokenizer, target_dict)
            #model = Rationale_With_Labels(768, 2, params).to(device)
        else:    
            train_data,valid_data,test_data,_=load_data_own_new(data_path=params['dataset'],number_of_samples=params['training_points'], random_seed=params['data_seed'])
            
            params['num_classes']=len(train_data['label'].unique())
            
            train_data_source = Normal_Dataset(train_data,params,tokenizer, train = True)
            val_data_source = Normal_Dataset(valid_data,params,tokenizer)
            test_data_source = Normal_Dataset(test_data,params,tokenizer)
            print(len(train_data_source.DataLoader),len(val_data_source.DataLoader),len(test_data_source.DataLoader))
#             model = Model_with_labels(768, 2, params).to(device)
#             train(train_data_source.DataLoader, val_data_source.DataLoader, test_data_source.DataLoader,model, params,run)
#         if(params['train_rationale']==True):
#             model = Model_Rational_Label_SSDA.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)
#         elif(params['train_att']==True):
#             model = Model_Attention_Label_SSDA.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)
#         else:
#             model = Model_Label_SSDA.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)

    
        if(params['train_rationale']==True):
            model = Model_Rational_Label.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)
        elif(params['train_att']==True):
            model = Model_Attention_Label.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)
        else:
            model = Model_Label.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)
        
        train(train_data_source.DataLoader, val_data_source.DataLoader,test_data_source.DataLoader,model,tokenizer,params,run)
        

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Initialize parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('dataset',
                           metavar='--d',
                           type=str,
                           help='The dataset to not consider')
    
#     my_parser.add_argument('index',
#                            metavar='--i',
#                            type=int,
#                            help='list id to be used')
    
    my_parser.add_argument('gpuid',
                           metavar='--i',
                           type=int,
                           help='gpu id to be used')
    
    
    
    args = my_parser.parse_args()
    
#     with open(args.path,mode='r') as f:
#             params_list = json.load(f)
    
#     file_done='Done/'+args.path.split('/')[1][:-5]+'_done.json'
#     with open(file_done,mode='r') as f:
#          list_done = json.load(f)
#     if(args.index in list_done):
#         exit()
    
    RANDOM_SEED = params['data_seed']
    random_seed(RANDOM_SEED, True)
    
    
    
    params['dataset_original']=args.dataset
    #params=params_list[args.index]
    params['save_path']='Saved_Models/Best_Toxic_BERT/'   
    params['logging']='local'
    
#     params['learning_rate_bert']=params['learning_rate']
#     params['learning_rate_linear']=params['learning_rate']
#     params['entropy_lambda']=1
#     params['cache_path']='../Saved_models/'
    print(params)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        ##### You can set the device manually if you have only one gpu
        ##### comment this line if you don't want to manually set the gpu
#         deviceID = get_gpu()
#         torch.cuda.set_device(deviceID[0])
        ##### comment this line if you want to manually set the gpu
        #### required parameter is the gpu id
        torch.cuda.set_device(args.gpuid)
#        torch.cuda.set_device(0)
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
        params['iterations']=args.index
        params['file']=args.path
        
        run["parameters"] = params
        run["sys/tags"].add('Final_run_ICWSM')
        
    
    if(params['train_att']==params['train_rationale']==True):
        print("Cannot train both attention and NER rationale choose one ")
    else:
        
        for dataset in ['Davidson','Waseem','Founta','Basile', 'Olid']:
            if(dataset!=params['dataset_original']):
                train_caller(params,dataset,run)
                params['model_path']='Saved_Models/Best_Toxic_BERT/'+params['dataset_original']+'_all_but_one_BERT_toxic_label'
        
        
        
    if(run is not None):
        run.stop()
    
    