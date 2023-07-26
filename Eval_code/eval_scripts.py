import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score, classification_report
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification
import torch.nn as nn

def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat

def evaluate_classifier_and_return_predictions(test_dataloader, params, model, device):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])
    
    results_dict = {'results': []}
#     results_dict = {'sentences': [], 'correct_labels': [], 'predicted': []}
    total = 0
    correct = 0
    pred = []
    label = []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False, cache_dir=params['cache_path'])
#     criterion = nn.CrossEntropyLoss()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        b_input_ids, b_input_mask, b_attn, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device).long()
        
#         for input_ids in b_input_ids:
#             sentence = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
#             results_dict['sentences'].append(sentence)
        
#         for label in b_labels:
#             results_dict['correct_labels'].append(label.tolist())

        with torch.no_grad():        
            if(params['train_rationale']==True):
                ypred, _, _ = model(b_input_ids, b_input_mask, attn=b_attn,labels=b_labels)
            elif(params['train_att']==True):
                ypred, _ = model(b_input_ids, b_input_mask, attn=b_attn,labels=b_labels)
            else:    
                if type(model) == BertForSequenceClassification:
                    output = model(b_input_ids, b_input_mask)
                    output = output.logits
                else:
                    output = model(b_input_ids, b_input_mask, b_attn)
                if type(output) == tuple:
                    ypred, loss = output[0], output[1]
                else:
                    ypred = output
                    
#                     loss = criterion(ypred, b_labels)
#         print("Ypredshape == ", ypred.shape, "\n\n")
#         print(ypred)
        softmax = nn.Softmax(dim=1)
        ypred_softmaxed = softmax(ypred)
#         print("YpredSoftmaxedshape == ", ypred_softmaxed.shape, "\n\n")
#         print(ypred_softmaxed)
        
        
        for i in range(len(b_input_ids)):
            input_ids = b_input_ids[i]
            sentence = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True))
            label = b_labels[i]
            ypred_each = ypred_softmaxed[i]
            store_res_dict = {}
            results_here_dict = {'sentence': sentence, 'correct_label': label.tolist(), 'predicted': ypred_each.tolist()}
            results_dict['results'].append(results_here_dict)
            
        
#         for each in ypred_softmaxed:
#             results_dict['predicted'].append(each.tolist())

        ypred = ypred.cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        try:
            y_preds = np.hstack((y_preds, get_predicted(ypred)))
            y_test = np.hstack((y_test, label_ids))
        except:
            y_preds, y_test = ypred, label_ids
    
#     print(classification_report(y_test, y_preds))
#     f1 = f1_score(y_test, y_preds, average = 'macro')

    testf1=f1_score(y_test, y_preds, average='macro')
    testacc=accuracy_score(y_test,y_preds)
    testprecision=precision_score(y_test, y_preds, average='macro')
    testrecall=recall_score(y_test, y_preds, average='macro')
    cls_report = classification_report(y_test, y_preds, output_dict=True)
    return testf1,testacc,testprecision,testrecall, results_dict, cls_report 



def evaluate_classifier(test_dataloader, params,model,device):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])

    total = 0
    correct = 0
    pred = []
    label = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
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

        with torch.no_grad():        
            if(params['train_rationale']==True and params['train_targets']==True):
                ypred, _, logits, loss = model(b_input_ids, b_input_mask, attn=b_attn,targets=b_targets,labels=b_labels)
            elif(params['train_rationale']==True and params['train_targets']==False):
                ypred, _, logits, loss = model(b_input_ids, b_input_mask, attn=b_attn,targets=None,labels=b_labels)           
            elif(params['train_att']==True):
                ypred, _ = model(b_input_ids, b_input_mask, attn=b_attn,labels=b_labels)
            else:    
                ypred, _ = model(b_input_ids, b_input_mask,labels=b_labels)

        ypred = ypred.cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        try:
            y_preds = np.hstack((y_preds, get_predicted(ypred)))
            y_test = np.hstack((y_test, label_ids))
        except:
            y_preds, y_test = ypred, label_ids
    
#     print(classification_report(y_test, y_preds))
#     f1 = f1_score(y_test, y_preds, average = 'macro')

    testf1=f1_score(y_test, y_preds, average='macro')
    testacc=accuracy_score(y_test,y_preds)
    testprecision=precision_score(y_test, y_preds, average='macro')
    testrecall=recall_score(y_test, y_preds, average='macro')
    return testf1,testacc,testprecision,testrecall 



def evaluate_target_classifier(test_dataloader, params,model,device):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])

    total = 0
    correct = 0
    pred = []
    label = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        b_input_ids=batch[0].to(device) 
        b_input_mask=batch[1].to(device)
        b_attn = batch[2].to(device).long()
        b_targets = batch[3].to(device)
        b_labels = batch[4].to(device).long()
        with torch.no_grad():        
            if(params['train_rationale']==True and params['train_targets']==True):
                _, ypred_target, _ ,_ = model(b_input_ids, b_input_mask, attn=b_attn,targets=b_targets,labels=b_labels)
            elif(params['train_rationale']==True and params['train_targets']==False):
                _, ypred_target, _ ,_ = model(b_input_ids, b_input_mask, attn=b_attn,targets=None,labels=b_labels)           
            elif(params['train_att']==True):
                ypred, _ = model(b_input_ids, b_input_mask, attn=b_attn,labels=b_labels)
            else:    
                ypred, _ = model(b_input_ids, b_input_mask,labels=b_labels)

        ypred_target = ypred_target.cpu().numpy()
        b_targets = b_targets.to('cpu').numpy()
        try:
            y_preds = np.hstack((y_preds, ypred_target))
            y_test = np.hstack((y_test, b_targets))
        except:
            y_preds, y_test = ypred_target, b_targets
    
#     print(classification_report(y_test, y_preds))
#     f1 = f1_score(y_test, y_preds, average = 'macro')
    y_preds = np.array(y_preds) >= 0.5
    testf1=f1_score(y_test, y_preds, average='macro')
    testacc=accuracy_score(y_test,y_preds)
    testprecision=precision_score(y_test, y_preds, average='macro')
    testrecall=recall_score(y_test, y_preds, average='macro')
    return testf1,testacc,testprecision,testrecall 


def evaluate_rationales(test_dataloader,params, model,device):
    model.eval()
    y_preds, y_test, y_mask = None, None, None

    total = 0
    correct = 0
    pred = []
    label = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        b_input_ids=batch[0].to(device) 
        b_input_mask=batch[1].to(device)
        b_attn = batch[2].to(device).long()
        b_targets = batch[3].to(device)
        b_labels = batch[4].to(device).long()
        with torch.no_grad():
            if(params['train_rationale']==True and params['train_targets']==True):
                _,_,logits, _ = model(b_input_ids, b_input_mask, attn=b_attn,targets=b_targets,labels=b_labels)
            elif(params['train_rationale']==True and params['train_targets']==False):
                _,_,logits, _ = model(b_input_ids, b_input_mask, attn=b_attn,targets=None,labels=b_labels)
            
            else:
                print("Rationale cannot be evaluated")
        
        ypred = logits.cpu().numpy()
        label_ids = b_attn.to('cpu').numpy()
        mask = b_input_mask.to('cpu').numpy()
        try:
            y_preds = np.hstack((y_preds, ypred))
            y_test = np.hstack((y_test, label_ids))
            y_mask = np.hstack((y_mask, mask))
        except:
            y_preds, y_test, y_mask = ypred, label_ids, mask

    for i in range(y_mask.shape[0]):
        for j in range(len(y_mask[i])):
            # if y_mask[i][j] == 0: break
            if np.argmax(y_preds[i][j]) == y_test[i][j]: correct += 1
            pred.append(np.argmax(y_preds[i][j]))
            label.append(y_test[i][j])
            total += 1
    
    
    testf1=f1_score(pred, label, average='macro')
    testacc=accuracy_score(pred,label)
    testprecision=precision_score(pred, label, average='binary')
    testrecall=recall_score(pred, label, average='binary')    
    return testf1,testacc,testprecision,testrecall 



#     acc = correct/total
#     print(classification_report(label, pred))
#     return (acc, correct, total, pred, label)