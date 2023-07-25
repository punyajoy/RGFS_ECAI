import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
import os 
import random
import numpy as np

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)



class Predictor(nn.Module):
    def __init__(self,hidden_size=4096,num_class=64):
        super(Predictor, self).__init__()
        self.num_class = num_class
        self.fc = nn.Linear(hidden_size, num_class, bias=False)
        self.temp = 0.05

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


def cross_entropy(input1, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * logsoftmax(input1))
  

def masked_cross_entropy(input1,target,mask):
    cr_ent=0
    for h in range(0,mask.shape[0]):
        cr_ent+=cross_entropy(input1[h][mask[h]],target[h][mask[h]])
    
    return cr_ent/mask.shape[0]



class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
def save_bert_model(model,tokenizer,params):
        output_dir = params['save_path']
        str_add = ""
        if(params['train_att']==True):
            if(params['attn_lambda']>=1):
                params['attn_lambda']=int(params['attn_lambda'])
            str_add +='_attention_'+ str(params['attn_lambda'])
        if(params['train_rationale']==True):
            if(params['rationale_impact']>=1):
                params['rationale_impact']=int(params['rationale_impact'])
            str_add +='_rationale_'+ str(params['rationale_impact'])
            
        if(params['train_targets']==True):
            if(params['target_impact']>=1):
                params['target_impact']=int(params['target_impact'])
            str_add +='_target_'+ str(params['target_impact'])
        else:
            output_dir =  output_dir+params['dataset']+'_BERT_toxic_label'+'/'
        
        if(str_add!=""):
            output_dir =  output_dir+'BERT_toxic'+str_add+'/'
        
        print(output_dir)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        
        
def save_bert_model_only(model,tokenizer,params):
        output_dir = params['save_path']
        if(params['train_att']==True):
            if(params['attn_lambda']>=1):
                params['attn_lambda']=int(params['attn_lambda'])
            output_dir =  output_dir+'BERT_toxic_attn_ab_'+str(params['attn_lambda'])+'/'
            
        elif(params['train_rationale']==True):
            if(params['rationale_impact']>=1):
                params['rationale_impact']=int(params['rationale_impact'])
            output_dir =  output_dir+'BERT_toxic_rationale_ab_'+str(params['rationale_impact'])+'/'
        elif(params['train_targets']==True):
            if(params['target_impact']>=1):
                params['target_impact']=int(params['target_impact'])
            output_dir =  output_dir+'BERT_toxic_target_ab_'+str(params['target_impact'])+'/'
        else:
            output_dir =  output_dir+params['dataset']+'_BERT_toxic_label_ab'+'/'
        print(output_dir)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model=model.bert
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        
def save_bert_model_all_but_one(model,tokenizer,params):
        output_dir = params['save_path']
        if(params['train_att']==True):
            if(params['attn_lambda']>=1):
                params['attn_lambda']=int(params['attn_lambda'])
            output_dir =  output_dir+'BERT_toxic_attn_ab_'+str(params['attn_lambda'])+'/'
            
        elif(params['train_rationale']==True):
            if(params['rationale_impact']>=1):
                params['rationale_impact']=int(params['rationale_impact'])
            output_dir =  output_dir+'BERT_toxic_rationale_ab_'+str(params['rationale_impact'])+'/'
        elif(params['train_targets']==True):
            if(params['target_impact']>=1):
                params['target_impact']=int(params['target_impact'])
            output_dir =  output_dir+'BERT_toxic_target_ab_'+str(params['target_impact'])+'/'
        else:
            output_dir =  output_dir+params['dataset_original']+'_all_but_one_BERT_toxic_label/'
        print(output_dir)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model=model.bert
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def save_bert_model_plus(model,tokenizer,params):
        output_dir = params['save_path']
        
        output_dir =  output_dir+model_name+'_'+params['dataset']+'/'
        print(output_dir)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        
def save_bert_model_fewshot(model,tokenizer,params):
    output_dir = params['save_path']+params['dataset']+'/'
    model_name = params['model_path'].split('/')[-1]
    output_dir =  output_dir+model_name+'_'+str(params['training_points'])+'_'+str(params['data_seed'])+'/'
    print(output_dir)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

