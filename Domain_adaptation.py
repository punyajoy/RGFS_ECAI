import numpy as np
from Data_code.data import *
from Data_code.load_data import *
from Model_code.models import *
from Model_code.utils import *
from Eval_code.eval_scripts import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
import neptune.new as neptune
import numpy as np
from datasets import list_datasets, load_dataset
import pandas as pd
from tqdm import tqdm
import argparse
import json
import time
import datetime
import pprint


#classes in the different datasets
classes_in_dataset = {
    'Davidson': 3,
    'Founta': 3,
    'Basile': 2,
    'Olid': 2,
    'Waseem':2
}


#the parameters dict with some of the default values other values will be loaded at the command line.
params={
 'dataset': None,
 'training_points': None, # CHANGE
 'model_path': 'bert-base-uncased',        # CHANGE
 'model_type': None,      # CHANGE
 'training_type': 'normal',
 'logging': 'local',
 'data_seed': None,
 'cache_path':'../../Saved_models/',
 'learning_rate':1e-5,
 'device':'cuda',
 'num_classes': -1,
 'epochs': 20,
 'batch_size':16,
 'rationale_impact': 10,
 'attn_lambda':1,
 'train_rationale':False,
 'train_att':False,
 'save_model': False,
 'save_path':'Saved_Models/Final_AAAI2023',
 'predictions_save_path': 'Model_Predictions/Final_AAAI2023/NonRationales/',
 'random_rationales': False,
 'non_rationales': False,
}

# set random seed for reproducibility
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Converts time taken for training to the hh:mm:ss format
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# trainig for the few shot setting
def train(training_dataloader, validation_dataloader, test_dataloader, model, tokenizer, params,run):
    total_t0 = time.time()
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
    best_macro_f1_test = 0
    best_accuracy_test = 0
    best_pre_test = 0
    best_rec_test = 0
    best_cls_report_val = None
    best_cls_report_test = None
    
    best_macro_f1_rat_test = 0
    best_acc_rat_test =0
    best_pre_rat_test = 0
    best_rec_rat_test = 0
    
    best_model = None

    criterion = nn.CrossEntropyLoss()
    
    # Sets model save folder as params['save_path']/params['dataset']/model_name
    # str(class), eg. "<class '__main__.abc'>" to "abc" below line
    if params['save_model'] == True:
        model_name = str(params['model_type']).split('.')[2].split("'")[0]
        print("\n\nMODEL NAME = ", model_name, "\n\n")
        if params['random_rationales'] == True:
              model_name = model_name + "_RANDOM"
        model_save_path = os.path.join(params['save_path'], params['dataset'], model_name)
        if os.path.isdir(model_save_path) == True:
            i = 1
            model_save_path = model_save_path + str(i)
            while os.path.isdir(model_save_path) != False:
                i += 1
                model_save_path = model_save_path[-1] + str(i)
        os.makedirs(model_save_path)
        print("Model Save Path = " + str(model_save_path))
    
    
    print("Running Few-Shot Training on " + str(params['training_points']) + " datapoints of the " + str(params['dataset']) + " dataset...")
    for epoch_i in range(0, epochs):
        print("Epoch " + str(epoch_i + 1))
        model.train()
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            b_input_ids=batch[0].to(device) 
            b_input_mask=batch[1].to(device)
            b_attn = batch[2].to(device)
            b_labels = batch[3].to(device).long()
            if type(model) == BertForSequenceClassification:
                #for baseline models 
                output = model(b_input_ids, b_input_mask)
                output = output.logits
            else:
                #for RGFS models
                output = model(b_input_ids, b_input_mask, b_attn)

            if type(output) == tuple:
                ypred = output[0]
                loss = output[1]
            else:
                ypred = output
                loss = criterion(ypred, b_labels)
            # print(outputs.logits)
            if(params['logging']=='local'):
                if step%100 == 0:
                    print(loss.item())
           
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    
        print("Running Eval on Train Data...")
        # Uncomment if you want to run evaluation on the training datasets
        # BEWARE: This will take a long time
#         macro_f1_train,accuracy_train, pre_train, rec_train, results_dict_train, cls_report_train = evaluate_classifier_and_return_predictions(training_dataloader, params,model,device)
        print("Running Validation...")
        macro_f1_val,accuracy_val, pre_val, rec_val, results_dict_val, cls_report_val = evaluate_classifier_and_return_predictions(validation_dataloader, params,model,device)
        if macro_f1_val > best_macro_f1_val:
            print("Running Test...")
            macro_f1_test,accuracy_test, pre_test, rec_test, results_dict_test, cls_report_test = evaluate_classifier_and_return_predictions(test_dataloader, params,model,device)
        
        
#         Uncomment if you want to run evaluation on the training datasets    
#         print("  CLS Report Train: " + str(cls_report_train))
#         print("  Macro F1 Train: {0:.3f}".format(cls_report_train['macro avg']['f1-score']))
        print("  Macro F1 Val: {0:.3f}".format(macro_f1_val))
        print("  Macro F1 Test: {0:.3f}".format(macro_f1_test))
        print("  CLS Report Val: " + str(cls_report_val) + "\n")
        
        if macro_f1_val > best_macro_f1_val:
#             Uncomment the line below if you are running evaluation on the training datasets
#             best_results_dict_train = results_dict_train
            best_results_dict_val = results_dict_val
            best_results_dict_test = results_dict_test
            
            best_macro_f1_val = macro_f1_val
            best_macro_f1_test = macro_f1_test
            best_accuracy_test = accuracy_test
            best_pre_test = pre_test
            best_rec_test = rec_test
#             Uncomment the line below if you are running evaluation on the training datasets
#             best_cls_report_train = cls_report_train
            best_cls_report_val = cls_report_val
            best_cls_report_test = cls_report_test
            
               
            if params['save_model'] == True:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'model_weights.pt'))


#     Uncomment the line below if you are running evaluation on the training datasets

#     print("\nBest Macro F1 Train = " + str(best_cls_report_train['macro avg']['f1-score']))
    print("Best Macro-F1 Val = " + str(best_macro_f1_val))
    print("Best Macro-F1 Test = " + str(best_macro_f1_test))
    print("Best Precision Test = " + str(best_pre_test))
    print("Best Recall Test = " + str(best_rec_test)) 
    print("Best Accuracy Test = " + str(best_accuracy_test))
#     print("Best CLS Report Train = " + str(best_cls_report_train))
    print("Best CLS Report Val = " + str(best_cls_report_val))
    print("Best CLS Report Test = " + str(best_cls_report_test))
    if params['train_rationale'] == True:
        print("Best Macro-F1 Rationale Test = " + str(best_macro_f1_rat_test))
        print("Best Precision Rationale Test = " + str(best_pre_rat_test))
        print("Best Recall Rationale Test = " + str(best_rec_rat_test))
        print("Best Accuracy Rationale Test = " + str(best_acc_rat_test) + "\n")
    
 
    # Saves run info, model predictions in params['predictions_save_path']/{}TP_{}RS_{HXTrained}_{ModelName}, eg:
    # params['predictions_save_path']/50TP_2021RS_HXTrained_Transform_Rationale_Mask
    # From params: TP-Training Points, RS-Random State, HX Trained-whether params['model_path'] == 'bert-base-uncased' or other
    model_name = str(params['model_type']).split('.')[2].split("'")[0]
    if params['model_path'] == 'bert-base-uncased':
        save_folder_name = str(params['training_points']) + "TP_" + str(params['data_seed']) + "RS_" + model_name
    else:
        save_folder_name = str(params['training_points']) + "TP_" + str(params['data_seed']) + "RS_" + model_name + "_HXTrained"
    if params['random_rationales'] == True:
        save_folder_name = save_folder_name + "_RANDOM"
    save_dir_path = os.path.join(params['predictions_save_path'], save_folder_name)
    if os.path.isdir(save_dir_path):
        i = 1
        save_dir_path = save_dir_path + "_1"
        while(os.path.isdir(save_dir_path) != False):
            i += 1
            save_dir_path = save_dir_path[:-1] + str(i)
    os.makedirs(save_dir_path)
    print("Saving run info and predictions on the val and test sets at " + save_dir_path + "...")
    
    with open(os.path.join(save_dir_path, 'run_info.txt'), 'w') as f1:
        params_copy = params
        params_copy['model_type'] = str(params_copy['model_type'])
#         Uncomment the line below if you are running evaluation on the training datasets
#         f1.write("Best Macro-F1 Train = " + str(best_cls_report_train['macro avg']['f1-score']) + "\n")
        f1.write("Best Macro-F1 Val = " + str(best_macro_f1_val) + "\n")
        f1.write("Best Macro-F1 Test = " + str(best_macro_f1_test) + "\n")
        f1.write("Best Precision Test = " + str(best_pre_test) + "\n")
        f1.write("Best Recall Test = " + str(best_rec_test) + "\n")
        f1.write("Best Accuracy Test = " + str(best_accuracy_test) + "\n")
#         Uncomment the line below if you are running evaluation on the training datasets
#         f1.write("\nBest CLS Report Train = " + str(best_cls_report_train) + "\n")
        f1.write("Best CLS Report Val = " + str(best_cls_report_val) + "\n")
        f1.write("Best CLS Report Test = " + str(best_cls_report_test) + "\n\n")
        if params['train_rationale'] == True:
            f1.write("Best Macro-F1 Rationale Test = " + str(best_macro_f1_rat_test) + "\n")
            f1.write("Best Precision Rationale Test = " + str(best_pre_rat_test) + "\n")
            f1.write("Best Recall Rationale Test = " + str(best_rec_rat_test) + "\n")
            f1.write("Best Accuracy Rationale Test = " + str(best_acc_rat_test) + "\n\n")
        if params['random_rationales'] == False:
            params['model_type'] = str(params['model_type'])
        else:
            params['model_type'] = str(params['model_type'])[:-2] + "_RANDOM" + str(params['model_type'][-2:])
        f1.write(json.dumps(params_copy, indent=4))

#     Uncomment the line below if you are running evaluation on the training datasets        
#     save_path_train = os.path.join(save_dir_path, 'preds_train.json')
#     with open(save_path_train, 'w') as f1:
#         json.dump(best_results_dict_train, f1, indent=4)

    save_path_val = os.path.join(save_dir_path, 'preds_val.json')
    with open(save_path_val, 'w') as f1:
        json.dump(best_results_dict_val, f1, indent=4)

    save_path_test = os.path.join(save_dir_path, 'preds_test.json')
    with open(save_path_test, 'w') as f1:
        json.dump(best_results_dict_test, f1, indent=4)
    print("\nRandom Seed = " + str(params['data_seed']))
    print("Model Path = " + str(params['model_path']))
    print("Model Type = " + str(params['model_type']))
    print("Best Macro F1 Test = " + str(best_macro_f1_test))
    print("Training Points = " + str(params['training_points']))
    print("Done!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    print("\n\n")
            
            

def train_caller(params):
    if(params['training_type']=='normal'):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
       
        train_data,valid_data,test_data,_=load_data_own_new(data_path=params['dataset'],number_of_samples=params['training_points'], random_seed=params['data_seed'])
        
#             print("Loading the rationale predictor")
# BEST RATIONALE PREDICTOR MODEL = BERT_toxic_rationale_2

#             rationale_predictor_model = Model_Rational_Label.from_pretrained("Saved_Models/Best_Toxic_BERT/BERT_toxic_rationale_10", params={'num_classes':3, 'rationale_impact':10,'target_impact':10,'targets_num':22},output_attentions = True,output_hidden_states = False).to(params['device'])
        rat_start_time = time.time()
        rationale_predictor_model = Model_Rational_Label.from_pretrained("Saved_Models/Best_Toxic_BERT/BERT_toxic_rationale_target_2", params={'num_classes':2, 'rationale_impact':10,'target_impact':10,'targets_num':22},output_attentions = True,output_hidden_states = False).to(params['device'])

        rationale_predictor = modelPred(params=params, model=rationale_predictor_model)
        
        print("Predicting rationales...")
        train_data_source = Normal_Dataset_new(train_data, rationale_predictor, params,tokenizer, train = True)
        val_data_source = Normal_Dataset_new(valid_data, rationale_predictor, params, tokenizer)
        test_data_source = Normal_Dataset_new(test_data, rationale_predictor, params, tokenizer)
        print(len(train_data_source.DataLoader),len(val_data_source.DataLoader),len(test_data_source.DataLoader))
        print(f"RATIONALE PREDICTION TIME = {time.time() - rat_start_time}")

        if params['model_path'] == 'bert-base-uncased':
            if params['model_type'] == BertForSequenceClassification:
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=params['num_classes']).to(device)
            else:
                model = params['model_type'].from_pretrained(params['model_path'], params=params).to(device)
        else:
            print("Loading pretrained model from " + str(params['model_path']) + "...")
            pretrained_dict = torch.load(params['model_path'])
            
            if params['model_type'] == BertForSequenceClassification:
                config = BertConfig()
                config.num_labels = params['num_classes']
                model = BertForSequenceClassification(config).to(device)
            else:
                config = BertConfig()
                model = params['model_type'](config=config, params=params).to(device)
    
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            print("Model Loaded!")
            
        
        
        
        train(train_data_source.DataLoader, val_data_source.DataLoader,test_data_source.DataLoader,model,tokenizer,params)
        
        
if __name__ == "__main__":
    # Initialize parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('dataset',
                           metavar='-d',
                           type=str,
                           help='The name of the training dataset to be used')
    
    my_parser.add_argument('training_points',
                           metavar='-tp',
                           type=int,
                           help='The number of training points for the few-shot training dataset')
    
    my_parser.add_argument('random_seed',
                           metavar='-rs',
                           type=int,
                           help='The random state to be used')
    
    my_parser.add_argument('model',
                           metavar='-m',
                           type=int,
                           help='The number (index) of the model to be used')

    my_parser.add_argument('random_rationales',
                           metavar='-r',
                           type=int,
                           help='Whether (1) or not (0) to use random rationales')

    
    args = my_parser.parse_args()
    # print(args)
    index_to_model = {
        1: BertForSequenceClassification, # IMPORTANT
        16: Transform_Rationale_CrossAttn_CLS_Drpt_corrected, # IMPORTANT
        17: Transform_Rationale_SelfAttn_Drpt_corrected, # IMPORTANT
    }
    
    params['dataset'] = args.dataset
    params['num_classes'] = classes_in_dataset[params['dataset']]
    params['predictions_save_path'] = os.path.join(params['predictions_save_path'], params['dataset'])
    params['training_points'] = args.training_points
    params['data_seed'] = args.random_seed
    RANDOM_SEED = params['data_seed']
    random_seed(RANDOM_SEED, True)

    params['model_type'] = index_to_model[args.model]

    params['random_rationales'] = False if args.random_rationales == 0 else True

    pprint.pprint(params)

    if torch.cuda.is_available() and params['device']=='cuda':     
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("Using GPU " + str(0))

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")    
    