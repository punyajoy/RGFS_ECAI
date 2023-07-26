import torch 
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
from .utils import *
import torch.nn.functional as F

#
from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
# model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")



class Model_Rational_Label(BertPreTrainedModel):
     def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.impact_factor=params['rationale_impact']
        self.bert = BertModel(config,add_pooling_layer=False)
        self.bert_pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()        
#         self.embeddings = AutoModelForTokenClassification.from_pretrained(params['model_path'], cache_dir=params['cache_path'])
        
     def forward(self, input_ids=None, mask=None, attn=None, labels=None):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
#         mean_pooling = torch.mean(out, 1)
#         max_pooling, _ = torch.max(out, 1)
#         embed = torch.cat((mean_pooling, max_pooling), 1)
        embed=self.bert_pooler(outputs[0])
        y_pred = self.classifier(self.dropout(embed))
        loss_token = None
        loss_label = None
        loss_total = None
        
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                )
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total=self.impact_factor*loss_token
            
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if(loss_total is not None):
                loss_total+=loss_label
            else:
                loss_total=loss_label
        if(loss_total is not None):
            return y_pred, logits, loss_total
        else:
            return y_pred, logits

        
        
#
class Model_Rationale_Label_New(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.impact_factor=params['rationale_impact']
#         self.entropy_lambda=params['entropy_lambda']
#         self.bert = BertModel(config,add_pooling_layer=False)
#         self.bert_pooler=BertPooler(config)

        self.hatexplain_trained_classifier = AutoModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        self.model_path = 'Saved_Models/Best_Toxic_BERT/BERT_toxic_rationale_10_random'
        self.rationale_predictor = AutoModel(model_path)#, add_pooling_layer=False)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
#         self.classifier = Predictor(config.hidden_size, self.num_labels)
        self.init_weights()
        
#         self.rationale_classifier_linear = nn.Linear(config.hidden_size, 2)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
    
    def forward(self, input_ids=None, mask=None, attn=None, labels=None):
        classifier_output = self.hatexplain_trained_classifier(input_ids, mask)[0]
        rationalePred_output = self.rationale_predictor(input_ids, mask)[0]
        
#         classifier_output = classifier_output[0]
#         rationale_output = rationalePred_output[0]        
        
        logits = self.token_classifier(self.token_dropout(rationalePred_output))
        
#         hidden_size = len(classifier_output[2])
#         batch_size = len(classifier_output[0])
        seq_length = len(classifier_output[1])
        
#         multiply_rationale_val = rationale_probs[:][:][1]
        for i in range(config.batch_size):
            for j in range(seq_length):
                classifier_output[i][j] *= rationale_probs[i][j][1] 
        
        res = torch.sum(classifier_output, dim=1)
        y_pred = self.classifier(self.dropout(res))
        
        loss_token = None
        loss_label = None
        loss_total = None
        
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                )
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total=self.impact_factor*loss_token
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if(loss_total is not None):
                loss_total+=loss_label
            else:
                loss_total=loss_label
        if(loss_total is not None):
            return y_pred, logits, loss_total
        else:
            return y_pred, logits
        
#         return rationale_probs, final_cls_probs
        

        
        
        
        
        
        
class Model_Label(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, attn=None, labels=None):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        
        pooled_output = outputs[1]
        y_pred = self.classifier(self.dropout(pooled_output))
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return y_pred, loss_label
        else:
            return y_pred
        
        
class Model_Attention_Label(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.lam = params['attn_lambda']
        self.num_sv_heads = 6
        self.sv_layer = 11
        self.train_att=params['train_att']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    
    def forward(self, input_ids=None, mask=None, attn=None, labels=None):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        pooled_output = outputs[1]
        y_pred = self.classifier(self.dropout(pooled_output))
        
        loss_label = None
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if attn is not None:
                loss_att=0
                for i in range(self.num_sv_heads):
                    attention_weights=outputs[2][self.sv_layer][:,i,0,:]
                    loss_att +=self.lam*masked_cross_entropy(attention_weights,attn,mask)
                loss_label = loss_label + loss_att
        
        
        if(loss_label is not None):
            return y_pred, loss_label
        else:
            return y_pred

        

        
class Model_Label_SSDA(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.entropy_lambda=params['entropy_lambda']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = Predictor(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, attn=None, labels=None, type1='label'):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        pooled_output = outputs[1]
        pooled_output=self.dropout(pooled_output)
        y_pred = self.classifier(pooled_output)
        loss_univ=None
        if(type1=='label'):
            if labels is not None:
                loss_funct = nn.CrossEntropyLoss()
                loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
                loss_univ= loss_logits
        elif(type1=='unlabel'):
            loss_univ = adentropy(self.classifier, pooled_output, self.entropy_lambda)
            
        if(loss_univ is not None):
            return y_pred, loss_univ
        else:
            return y_pred

        
        
        
class Model_Attention_Label_SSDA(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.entropy_lambda=params['entropy_lambda']
        self.lam = params['attn_lambda']
        self.num_sv_heads = 6
        self.sv_layer = 11
        self.train_att=params['train_att']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = Predictor(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, attn=None, labels=None, type1='label'):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        pooled_output = outputs[1]
        pooled_output=self.dropout(pooled_output)
        y_pred = self.classifier(pooled_output)
        loss_univ=None
        if(type1=='label'):
            if labels is not None:
                loss_funct = nn.CrossEntropyLoss()
                loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
                loss_label= loss_logits
                if(self.train_att):
                    loss_att=0
                    for i in range(self.num_sv_heads):
                        attention_weights=outputs[2][self.sv_layer][:,i,0,:]
                        loss_att +=self.lam*masked_cross_entropy(attention_weights,attn,mask)
                    loss_univ = loss_label + loss_att
                else:
                    loss_univ = loss_label
        elif(type1=='unlabel'):
            loss_univ = adentropy(self.classifier, pooled_output, self.entropy_lambda)
            
        if(loss_univ is not None):
            return y_pred, loss_univ
        else:
            return y_pred

        
        
class Model_Rational_Label_SSDA(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.impact_factor=params['rationale_impact']
        self.entropy_lambda=params['entropy_lambda']
        self.bert = BertModel(config,add_pooling_layer=False)
        self.bert_pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = Predictor(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, attn=None, labels=None, type1='label'):
        outputs = self.bert(input_ids, mask)
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
        pooled_output=self.bert_pooler(outputs[0])
        y_pred = self.classifier(self.dropout(pooled_output))
        
        # out = outputs.last_hidden_state
        
        loss_univ=None
        if(type1=='label'):
            if attn is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if mask is not None:
                    active_loss = mask.view(-1) == 1
                    active_logits = logits.view(-1, 2)
                    active_labels = torch.where(
                        active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                    )
                    loss_token = loss_fct(active_logits, active_labels)
                else:
                    loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))

                loss_univ=self.impact_factor*loss_token
             
            if labels is not None:
                loss_funct = nn.CrossEntropyLoss()
                loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
                loss_label= loss_logits
                if(loss_univ is not None):
                    loss_univ+=loss_label
                else:
                    loss_univ=loss_label
                    
        elif(type1=='unlabel'):
            loss_univ = adentropy(self.classifier, pooled_output, self.entropy_lambda)
            
        if(loss_univ is not None):
            return y_pred,logits,loss_univ
        else:
            return y_pred,logits

        
    
    
class Model_Rational_Label_SSDA_plus(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.impact_factor=params['rationale_impact']
        self.entropy_lambda=params['entropy_lambda']
        self.bert = BertModel(config,add_pooling_layer=False)
        self.bert_pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = Predictor(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = Predictor(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, attn=None, labels=None, type1='label'):
        outputs = self.bert(input_ids, mask)
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
        pooled_output=self.bert_pooler(outputs[0])
        y_pred = self.classifier(self.dropout(pooled_output))
        
        # out = outputs.last_hidden_state
        
        loss_univ=None
        if(type1=='label'):
            if attn is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if mask is not None:
                    active_loss = mask.view(-1) == 1
                    active_logits = logits.view(-1, 2)
                    active_labels = torch.where(
                        active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                    )
                    loss_token = loss_fct(active_logits, active_labels)
                else:
                    loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))

                loss_univ=self.impact_factor*loss_token
             
            if labels is not None:
                loss_funct = nn.CrossEntropyLoss()
                loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
                loss_label= loss_logits
                if(loss_univ is not None):
                    loss_univ+=loss_label
                else:
                    loss_univ=loss_label
                    
        elif(type1=='unlabel'):
            loss_univ = adentropy(self.classifier, pooled_output, self.entropy_lambda)
            loss_univ = loss_univ + adentropy(self.token_classifier, out, self.entropy_lambda)
            
        if(loss_univ is not None):
            return y_pred,logits,loss_univ
        else:
            return y_pred,logits

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    