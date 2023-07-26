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
import pandas as pd
from tqdm import tqdm
import argparse
import json
import time


params={
 'dataset':'Davidson',
 'training_points':-1,
 'model_path':'bert-base-uncased',
 'training_type':'normal',
 'logging':'local',
 'data_seed':2021,
 'cache_path':'../../Saved_models/',
 'learning_rate':1e-5,
 'device':'cuda',
 'num_classes':2,
 'epochs':10,
 'batch_size':16,
 'rationale_impact':100,
 'target_impact':0,
 'attn_lambda':1,
 'train_rationale':False,
 'train_targets':False,
 'train_att':False,
 'save_path':'Saved_Models/Best_Toxic_BERT/'   
}


# return the target dictionary for the hatexplain dataset
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
     
    return target_dict

    

def train(training_dataloader, validation_dataloader, test_dataloader, model, tokenizer, params):
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
    best_macro_f1_val_rat=0
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
            
        
        
        print("  Macro F1 Val: {0:.3f}".format(macro_f1_val))
        print("  Macro F1 Test: {0:.3f}".format(macro_f1_test))
        
        if(params['train_rationale']==True):
                print("  Macro F1 Rationale Val: {0:.3f}".format(macro_f1_val_rat))
                print("  Macro F1 Rationale Test: {0:.3f}".format(macro_f1_test_rat))
        
        if(params['train_targets']==True):
                print("  Macro F1 Target Val: {0:.3f}".format(macro_f1_val_tar))
                print("  Macro F1 Target Test: {0:.3f}".format(macro_f1_test_tar))
        
            
            
        #if (macro_f1_val > best_macro_f1_val) or (macro_f1_val_rat > best_macro_f1_val_rat):
        if (macro_f1_val_rat > best_macro_f1_val_rat):
            best_macro_f1_val_rat=macro_f1_val_rat
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


            save_bert_model(model,tokenizer,params)

################### Only for normal datasets
            #save_bert_model(model.bert,tokenizer,params)
#             best_model = copy.deepcopy(model)
#             save_metrics(filepath, epoch_i, model, optimizer, weighted_f1)
    
    print("  Macro F1 Test: {0:.3f}".format(best_macro_f1_test))
    print("  Accuracy Test: {0:.3f}".format(best_accuracy_test))

    if(params['train_rationale']==True):
        print("  Macro F1 Rationale Test: {0:.3f}".format(best_macro_f1_rat_test))
        print("  Accuracy Rationale Test: {0:.3f}".format(best_acc_rat_test))

    if(params['train_targets']==True):
        print("  Macro F1 Target Test: {0:.3f}".format(best_macro_f1_tar_test))
        print("  Accuracy Target Test: {0:.3f}".format(best_acc_tar_test))

            
            
            
            
            
            
            

def train_caller(params):
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
        else:    
            train_data,valid_data,test_data,_=load_data_own_new(data_path=params['dataset'],number_of_samples=params['training_points'], random_seed=params['data_seed'])
            
            params['num_classes']=len(train_data['label'].unique())
            
            train_data_source = Normal_Dataset(train_data,params,tokenizer, train = True)
            val_data_source = Normal_Dataset(valid_data,params,tokenizer)
            test_data_source = Normal_Dataset(test_data,params,tokenizer)
            print(len(train_data_source.DataLoader),len(val_data_source.DataLoader),len(test_data_source.DataLoader))
    
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
    my_parser.add_argument('path',
                           metavar='--p',
                           type=str,
                           help='The path to json containining the parameters')
    
    my_parser.add_argument('index',
                           metavar='--i',
                           type=int,
                           help='list id to be used')
    
    
    args = my_parser.parse_args()
    
    with open(args.path,mode='r') as f:
            params_list = json.load(f)
        
    RANDOM_SEED = params['data_seed']
    random_seed(RANDOM_SEED, True)
    
    
    
    
    params=params_list[args.index]
#     params['save_path']='Saved_Models/Best_Toxic_BERT/'   
    params['logging']='local'
    
    print(params)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    
    
    if(params['train_att']==params['train_rationale']==True):
        print("Cannot train both attention and NER rationale choose one ")
    else:
        train_caller(params)
    
    