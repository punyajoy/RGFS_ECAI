import torch 
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoModel
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
from .utils import *
import torch.nn.functional as F

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums

def masked_softmax_NormalizeLength(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums



class Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_basile(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_basile, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0

        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -1.2684).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -1.2684).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
    
    
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred


class Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_olid(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_olid, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0

        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.1782).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.1782).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
    
    
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred

class Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_davidson(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_davidson, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0

        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.0094).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.0094).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
    
    
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred

class Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_founta(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_founta, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0

        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.4646).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.4646).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
    
    
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred

class Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_waseem(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_random_nonrationales_waseem, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0

        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.3086).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.3086).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
    
    
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred




class Transform_Rationale_SelfAttn_Drpt_corrected(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred
    

class Transform_Rationale_SelfAttn_Drpt_corrected_clssep(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_clssep, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         i += 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
        for i in range(len(rationales_softmaxed)):
            rationales_softmaxed[i][mask[i].argmin() - 1] = 1
            rationales_softmaxed[i][0] = 1
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred
    
    
class Transform_Rationale_SelfAttn_Drpt_corrected_sigmoid(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_sigmoid, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_hate_class = torch.add(rationales[:, :, 1], 2)
#         rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        rationales_sigmoid = torch.sigmoid(rationales_hate_class.float())
#         i += 1
        for i in range(len(rationales_sigmoid)):
            rationales_sigmoid[i][mask[i].argmin() - 1] = 1
            rationales_sigmoid[i][0] = 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
#         print("rationales sigmoid shape = ", rationales_sigmoid.shape)
#         print("rationales sigmoid = ", rationales_sigmoid)
#         print("mask shape = ", mask.shape)
#         print("mask = ", mask)
        outputs = outputs * rationales_sigmoid.view(-1, 128, 1)
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred

class Transform_Rationale_SelfAttn_Drpt_corrected_scaled(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_scaled, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
#         i = 0
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_hate_class = torch.add(rationales[:, :, 1], 2)
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
#         rationales_sigmoid = torch.sigmoid(rationales_hate_class.float())
#         i += 1
        for i in range(len(rationales_softmaxed)):
            rationales_softmaxed[i][mask[i].argmin() - 1] = 1
            rationales_softmaxed[i][0] = 1
#         print(str(i) + "\t\t" + str(rationales_softmaxed) + "\n\n")
#         print("rationales sigmoid shape = ", rationales_sigmoid.shape)
#         print("rationales sigmoid = ", rationales_sigmoid)
#         print("mask shape = ", mask.shape)
#         print("mask = ", mask)
        outputs = outputs * rationales_softmaxed.view(-1, 128, 1)
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred    
    

    
class Transform_Rationale_SelfAttn_Drpt_corrected_NormalizeLength(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt_corrected_NormalizeLength, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax_NormalizeLength(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        outputs = outputs * rationales_softmaxed
        key_padding_mask = (mask == False)
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred


    
class Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_basile(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_basile, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        
        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -1.2684).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -1.2684).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
        
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

    
class Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_olid(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_olid, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        
        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.1782).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.1782).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
        
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

    
class Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_davidson(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_davidson, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        
        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.0094).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.0094).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
        
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred


class Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_founta(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_founta, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        
        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.4646).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.4646).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
        
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred
    

class Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_waseem(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected_random_nonrationales_waseem, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        
        ## SELECTS RATIONALES FROM NON-RATIONALES
        random_mask = (rationales[:, :, 1] <= -2.3086).float()
        random_tensor = ((torch.rand(rationales.shape[0], 128, 2) - 0.5) * 10).to('cuda')
        random_mask = random_mask.float() * random_tensor[:, :, 1][0]
        
        rationales_mask = ((rationales[:, :, 1] > -2.3086).long() * -4).to('cuda')
        
        new_rationales = rationales_mask + random_mask
#         print("\nOLD RATIONALES: ", rationales[:, :, 1][9][40:50])
#         print("\nNEW RATIONALES: ", new_rationales[9][40:50])
        rationales[:, :, 1] = new_rationales
        ## SELECTS RATIONALES FROM NON-RATIONALES
        
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Bert_C(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Bert_C, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        CLS_output = outputs.pooler_output
        y_pred = self.classifier(CLS_output)
        return y_pred

class Transform_Rationale_CrossAttn_CLS_Drpt_corrected(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Transform_Rationale_CrossAttn_CLS_Drpt_corrected_clssep(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_corrected_clssep, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = config.hidden_size
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        for i in range(len(rationales_softmaxed)):
            rationales_softmaxed[i][mask[i].argmin() - 1] = 1
            rationales_softmaxed[i][0] = 1
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        key_padding_mask = (mask == False)
        attn_result, _ = self.attn_with_CLS(q, k, v, key_padding_mask=key_padding_mask)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred    


class modelPred():
    def __init__(self, params, model):
        self.device = params['device']
        self.cache_path = params['cache_path']
        # self.model_type = self.process_path(model_path):
#         self.model = Model_Rational_Label.from_pretrained(\
#             model_path, params={'num_classes':2, 'rationale_impact':10},
#             output_attentions = True,output_hidden_states = False).to(self.device)
        self.model = model
        if self.device == 'cuda':
            self.model.cuda()  
        self.model.eval()
    
#     def process_path(self, model_path):
#         model_name = model_path.split('/')[3]
#         model_type = model_name.split('_')[2]
#         return model_type
        
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    def tokenize(self, sentences, padding = True, max_len = 128):
        input_ids, attention_masks, token_type_ids = [], [], []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir="Saved_models", local_files_only=False, force_download=False)
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {'input_ids': input_ids, 'attention_masks': attention_masks}
    
    def process_data(self, sentences_list):
        sentences = []
        for sentence in sentences_list:
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
        inputs = self.tokenize(sentences)
        return self.get_dataloader(inputs)
    
    def get_dataloader(self, inputs):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'])
        sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=32)
    
    def return_probab(self,sentences_list):
        """Input: should be a list of sentences"""
        """Output: probablity values"""
        device = self.device

        test_dataloader=self.process_data(sentences_list)

        print("Running the trained rationale predictor on the dataset...")
        logits_all=[]
        rationales_all = []
        # Evaluate data 
        # s = nn.Softmax(dim=2)
        for step,batch in enumerate(test_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
#             print("here1")
            outputs = self.model(b_input_ids, b_input_mask)
#             print("here2")
            if type(outputs) == tuple:
                logits = outputs[0]
                rationales = outputs[2]
            else:
                logits = outputs
                
            logits = logits.detach().cpu().numpy()
            rationales = rationales.detach().cpu().numpy()

            rationales_all+=list(rationales)
            logits_all+=list(logits)

#         logits_all_final=[]
#         rationales_all_final = [list(softmax(r)) for r in rationales_all]
#         for logits in logits_all:
#             logits_all_final.append(list(softmax(logits)))
#         print("Done")
        
        return np.array(logits_all), np.array(rationales_all)


class Transform_Bert_Manual(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Bert_Manual, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        CLS_output = outputs.pooler_output
        y_pred = self.classifier(CLS_output)
        return y_pred
    
class Transform_Rationale_Mask(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_Mask, self).__init__(config)
        self.criterion = nn.CrossEntropyLoss()
        self.num_labels = params['num_classes']
        self.softmax = nn.Softmax(dim=2)
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        outputs = outputs.last_hidden_state
        rationales = self.softmax(rationales)
        mask = rationales[:, :, 1].view(-1, 128, 1) > 0.5
        outputs = (outputs*mask).view(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred

class Transform_Rationale_Mask_2(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_Mask_2, self).__init__(config)
        self.criterion = nn.CrossEntropyLoss()
        self.num_labels = params['num_classes']
        self.softmax = nn.Softmax(dim=2)
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        outputs = outputs.last_hidden_state
        rationales = self.softmax(rationales)
        mask = rationales[:, :, 1].view(-1, 128, 1) > 0.2
        outputs = (outputs*mask).view(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred

class Transform_Rationale(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale, self).__init__(config)
        self.num_labels = params['num_classes']
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax(rationales)
        outputs = outputs.last_hidden_state*rationales[:, :, 1].view(-1, 128, 1)
        # outputs = torch.mean(outputs*mask, dim=1)/sum
        outputs = (outputs).view(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred

class Transform_Rationale_Mean(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_Mean, self).__init__(config)
        self.num_labels = params['num_classes']
        
        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax(rationales)
        outputs = outputs.last_hidden_state*rationales[:, :, 1].view(-1, 128, 1)
        outputs = torch.mean(outputs, dim=1)
        # outputs = (outputs).view(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred
    
class Transform_Mean(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Mean, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=2)
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        outputs = outputs.last_hidden_state
#         print("outputs shape = ", outputs.shape)
        outputs = torch.mean(outputs, dim=1)
        # outputs = (outputs).view(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred

class Transform_Rationale_SelfAttn(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
#         self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        outputs = outputs.last_hidden_state
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        outputs = outputs * rationales_softmaxed
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2))
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred

class Transform_Rationale_SelfAttn_2Softmax(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_2Softmax, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        outputs = outputs * rationales_softmaxed
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2))
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(outputs)
        return y_pred
    
class Transform_Rationale_SelfAttn_2Softmax_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_2Softmax_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        outputs = outputs * rationales_softmaxed
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2))
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred

class Transform_Rationale_SelfAttn_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_SelfAttn_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        outputs = outputs * rationales_softmaxed
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2))
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred


class Transform_NoRationale_SelfAttn_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_NoRationale_SelfAttn_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
#         self.embeddings = AutoModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.classifier = nn.Linear(768*128, self.num_labels)

    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        # outputs = outputs * rationales_softmaxed
        outputs, _ = self.attn(outputs.permute(1, 0, 2), outputs.permute(1, 0, 2), outputs.permute(1, 0, 2))
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 768*128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred
    

class Transform_Rationale_CrossAttn(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(128, self.num_labels)
        self.proj_128 = nn.Linear(768, 128)
        self.softmax = nn.Softmax(dim=1)
        self.embed_dim = 128
        self.num_heads = 8
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.init_weights()
        
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        outputs = outputs.last_hidden_state
        outputs = self.proj_128(outputs).permute(1, 0, 2)
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        q = rationales_softmaxed.permute(2, 0, 1)
        outputs, _ = self.attn(q, outputs, outputs)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 128)
        y_pred = self.classifier(outputs)
        return y_pred
    
class Transform_Rationale_CrossAttn_2Softmax(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_2Softmax, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(128, self.num_labels)
        self.proj_128 = nn.Linear(768, 128)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 128
        self.num_heads = 8
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.init_weights()
        
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        outputs = self.proj_128(outputs).permute(1, 0, 2)
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        q = rationales_softmaxed.permute(2, 0, 1)
        outputs, _ = self.attn(q, outputs, outputs)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 128)
        y_pred = self.classifier(outputs)
        return y_pred

class Transform_Rationale_CrossAttn_2Softmax_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_2Softmax_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.criterion = nn.CrossEntropyLoss()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(128, self.num_labels)
        self.proj_128 = nn.Linear(768, 128)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 128
        self.num_heads = 8
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
        self.init_weights()
        
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        outputs = outputs.last_hidden_state
        outputs = self.proj_128(self.dropout(outputs)).permute(1, 0, 2) # ADDED DROPOUT
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        q = rationales_softmaxed.permute(2, 0, 1)
        outputs, _ = self.attn(q, outputs, outputs)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1, 128)
        y_pred = self.classifier(self.dropout(outputs)) # ADDED DROPOUT
        return y_pred
    

# class Transform_Bert_SeqClass(BertPreTrainedModel):
#     def __init__(self, params):
#         super(Transform_Bert_SeqClass, self).__init__(config)
#         self.num_labels = params['num_classes']
#         self.bert = BertForSequenceClassification.from_pretrained(params['model_type'])
    
#     def forward(self, input_ids, mask, rationales):
#         y_pred = self.bert(input_ids, mask)
#         return y_pred
    
class Transform_Rationale_CrossAttn_CLS(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.embed_dim = 768
        self.num_heads = 12
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(attn_result.view(-1, 768))
        
        return y_pred
    
class Transform_Rationale_CrossAttn_CLS_2Softmax(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_2Softmax, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(attn_result.view(-1, 768))
        
        return y_pred

    
class Transform_Rationale_CrossAttn_CLS_2Softmax_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_2Softmax_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Transform_Rationale_CrossAttn_CLS_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
#         rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        rationales_softmaxed = masked_softmax(rationales[:, :, 1], mask, dim=1).view(-1, 128, 1)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Transform_Rationale_CrossAttn_CLS_Drpt_exp(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_exp, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        lhs_rationaleWeights = lhs * torch.exp(rationales_softmaxed) # torch.exp added here
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Transform_Rationale_CrossAttn_CLS_Drpt_2(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_2, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.4) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.4) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Transform_Rationale_CrossAttn_CLS_Drpt_3(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_Drpt_3, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.3) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.3) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
                
        return y_pred
        

class Transform_RandomRationales_CrossAttn_CLS_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_RandomRationales_CrossAttn_CLS_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.gpu = params['device']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        dim0 = int(rationales.shape[0]) # RANDOM
        random_rationales = -10 * torch.rand(dim0, 128, 1) + 5 # RANDOM
        random_rationales = random_rationales.to(self.gpu)
        rationales_softmaxed = self.softmax(random_rationales)
        lhs_rationaleWeights = lhs * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred


class Transform_NoRationale_CrossAttn_CLS_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_NoRationale_CrossAttn_CLS_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) # ADDED DROPOUT
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # ADDED DROPOUT
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        # rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        # rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
        # lhs_rationaleWeights = lhs * rationales_softmaxed
        lhs_rationaleWeights = lhs
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768))) # ADDED DROPOUT
        
        return y_pred

class Transform_Rationale_CrossAttn_CLS_2Softmax_Tuned(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_Rationale_CrossAttn_CLS_2Softmax_Tuned, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.classifier = nn.Linear(768, self.num_labels)
#         self.mask_level = nn.Linear(768, 1) # gets alpha - level (0 < alpha < 1) for masking lhs based on rationale scores
#         self.sigmoid = nn.Sigmoid() # sigmoid for alpha - (0 < alpha < 1)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2)
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2) # - dropout
        self.layernorm = nn.LayerNorm(768) # - layernorm
    
    def forward(self, input_ids, mask, rationales):
        outputs = self.bert(input_ids, mask)
        rationales = self.softmax2(rationales)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationales_softmaxed = self.softmax(rationales[:, :, 1].view(-1, 128, 1))
#         alpha = self.sigmoid(self.mask_level(CLS_output))
        rationale_mask = rationales_softmaxed > 0.25 # alpha is the rationale score over which each token's output is considered
        lhs_rationaleWeights = lhs * rationale_mask * rationales_softmaxed
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        attn_plus_CLS_result = CLS_output + attn_result.view(-1, 768) # - residual connection
        attn_plus_CLS_result = self.layernorm(attn_plus_CLS_result) # - layernorm
        y_pred = self.classifier(self.dropout(attn_plus_CLS_result.view(-1, 768))) # - dropout
        
        return y_pred

class Transform_SingleTransformer_Rationale_CrossAttn_CLS_Drpt(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_SingleTransformer_Rationale_CrossAttn_CLS_Drpt, self).__init__(config)
        self.num_labels = params['num_classes']
        self.bert = Model_Rational_Label_Modded.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", \
                                     params={'num_classes':2, 'rationale_impact':10},output_attentions = True,\
                                     output_hidden_states = False).to(params['device'])
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = nn.Dropout(0.2) 
        self.attn_with_CLS = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.2)
    
    def forward(self, input_ids, mask, rationales):
        
        outputs = self.bert(input_ids, mask)
        embeds, rationaleWeight = outputs[0], outputs[1]

        lhs = embeds.last_hidden_state
        CLS_output = embeds.pooler_output
        
        # rationaleWeight = self.bert.return_probab(lhs) # added
        # rationaleWeight = self.softmax(rationaleWeight)
        
        lhs_rationaleWeights = lhs * rationaleWeight
        
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn_with_CLS(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(self.dropout(attn_result.view(-1, 768)))
        
        return y_pred

    
    
# class Transform_RationaleLoss_CrossAttn(BertPreTrainedModel):
#     def __init__(self, config, params):
#         super(Transform_RationaleLoss_CrossAttn, self).__init__(config)
#         self.num_labels = params['num_classes']
#         self.bert = BertModel(config)

class Transform_RationaleLoss_CrossAttn_2Softmax(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_RationaleLoss_CrossAttn_2Softmax, self).__init__(config)
        self.params = params
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(self.config.hidden_size, 2)
        self.classifier = nn.Linear(128, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()
        self.proj_128 = nn.Linear(768, 128)
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)
        self.embed_dim = 128
        self.num_heads = 8
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        
    def forward(self, input_ids, mask, attn=None, labels=None):
        outputs = self.bert(input_ids, mask)
        lhs = outputs.last_hidden_state
#         CLS_output = outputs.pooler_output
        rationale_logits = self.token_classifier(self.token_dropout(lhs))
        rationale_logits_softm = self.softmax2(rationale_logits)
        rationale_1_logits_softm_seqlen = self.softmax1(rationale_logits[:, :, 1])
#         lhs_rationaleWeights = lhs * rationale_1_logits_softm_seqlen.view(-1, 128, 1)

        lhs_projected = self.proj_128(lhs)
        q = rationale_1_logits_softm_seqlen.view(-1, 128, 1).permute(2, 0, 1)
        k = lhs_projected.permute(1, 0, 2)
        v = lhs_projected.permute(1, 0, 2)
        attn_result, _ = self.attn(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(attn_result.view(-1, 128))
        
        
        loss_label = None
        loss_token = None
        loss_total = None
        
        logits = rationale_logits
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                ).long()
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total = self.params['rationale_impact'] * loss_token
            
        loss_logits = self.criterion(y_pred.view(-1, self.num_labels), labels.view(-1).long())
        
        loss_label = loss_logits
        
        if (loss_total is not None):
            loss_total += loss_label
        else:
            loss_total = loss_label
        
        
        return y_pred, logits, loss_total

class Transform_RationaleLoss_CrossAttn_CLS_2Softmax(BertPreTrainedModel):
    def __init__(self, config, params):
        super(Transform_RationaleLoss_CrossAttn_CLS_2Softmax, self).__init__(config)
        self.params = params
        self.num_labels = params['num_classes']
        self.bert = BertModel(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(self.config.hidden_size, 2)
        self.classifier = nn.Linear(768, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)
        self.embed_dim = 768
        self.num_heads = 12
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        
    def forward(self, input_ids, mask, attn=None, labels=None):
        outputs = self.bert(input_ids, mask)
        lhs = outputs.last_hidden_state
        CLS_output = outputs.pooler_output
        rationale_logits = self.token_classifier(self.token_dropout(lhs))
        rationale_logits_softm = self.softmax2(rationale_logits)
        rationale_1_logits_softm_seqlen = self.softmax1(rationale_logits[:, :, 1])
        lhs_rationaleWeights = lhs * rationale_1_logits_softm_seqlen.view(-1, 128, 1)
        
        q = CLS_output.view(-1, 768, 1).permute(2, 0, 1)
        k = lhs_rationaleWeights.permute(1, 0, 2)
        v = lhs_rationaleWeights.permute(1, 0, 2)
        attn_result, _ = self.attn(q, k, v)
        attn_result = attn_result.permute(1, 0, 2)
        y_pred = self.classifier(attn_result.view(-1, 768))
        
        
        loss_label = None
        loss_token = None
        loss_total = None
        
        logits = rationale_logits
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                ).long()
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total = self.params['rationale_impact'] * loss_token
            
        loss_logits = self.criterion(y_pred.view(-1, self.num_labels), labels.view(-1).long())
        
        loss_label = loss_logits
        
        if (loss_total is not None):
            loss_total += loss_label
        else:
            loss_total = loss_label
        
        
        return y_pred, logits, loss_total
        
        

class Transform_7(BertPreTrainedModel):
    def __init__(self, config,  params):
#         self.config = AutoConfig.from_pretrained(params['model_path'])
        super(Transform_7, self).__init__(config)
#         super().__init__(self.config)
        self.num_labels=params['num_classes']
        self.model_path = params['model_path']
        self.embeddings = BertModel(config)
#         self.model_path = 'Saved_Models/Best_Toxic_BERT/BERT_toxic_rationale_10'
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(self.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.softmax2 = nn.Softmax(dim=2)
        self.softmax1 = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(128, self.num_labels)
        self.embed_dim = 128
        self.num_heads = 8
        self.proj_128 = nn.Linear(768, 128)
        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.init_weights()
    
    def forward(self, input_ids=None, mask=None, attn=None, labels=None):
        embeds = self.embeddings(input_ids, mask)
        last_hidden_state = embeds.last_hidden_state

        rationale_logits = self.token_classifier(self.token_dropout(last_hidden_state))
        rationale_1_logits_softm_seqlen = self.softmax1(rationale_logits[:, :, 1])
        
#         output = last_hidden_state * rationale_1_logits_softm_seqlen.view(-1, 128, 1) # 16 x 128 x 768 times 16 x 128 x 1
        
        output_permuted = self.proj_128(last_hidden_state).permute(1, 0, 2) # seq_len x batch_size x embed_dim - 128 x 16 x 128
        rationale_1_logits_softm_seqlen_permuted = rationale_1_logits_softm_seqlen.view(-1, 128, 1).permute(2, 0, 1)
        
#         print(rationale_1_logits_softm_seqlen_permuted.shape)
        
        try:
            output = self.attn(rationale_1_logits_softm_seqlen_permuted, output_permuted, output_permuted)
        except:
            print(rationale_1_logits_softm_seqlen_permuted.shape)
            print(output_permuted.shape)
        output = output[0].permute(1, 0, 2)
        
#         print("outputshape = ", output.shape)
        y_pred = output.reshape(-1, 128)
#         print("ypredshape = ", y_pred.shape)
        try:
            y_pred = self.classifier(y_pred)
        except:
            print(y_pred.shape)
        
#         print("ypredshape = ", y_pred.shape)

#         print(y_pred.shape)
#         print("rationalelogitsshape = ", rationale_logits.shape)
        
        loss_label = None
        loss_token = None
        loss_total = None
        
        logits = rationale_logits
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
            
            loss_total=loss_token
            
        loss_logits = self.criterion(y_pred.view(-1, self.num_labels), labels.view(-1))
        
        loss_label = loss_logits
        
        if (loss_total is not None):
            loss_total += loss_label
        else:
            loss_total = loss_label

        return y_pred, loss_total

class Model_Rational_Label(BertPreTrainedModel):
     def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.num_targets=params['targets_num']
        self.impact_factor=params['rationale_impact']
        self.target_factor=params['target_impact']
        self.bert = BertModel(config,add_pooling_layer=False)
        self.pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.2)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.target_dropout = nn.Dropout(0.2)
        self.target_classifier = nn.Linear(config.hidden_size, self.num_targets)
        self.init_weights()        
#         self.embeddings = AutoModelForTokenClassification.from_pretrained(params['model_path'], cache_dir=params['cache_path'])
        
     def forward(self, input_ids=None, mask=None, attn=None, labels=None, targets=None):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
#         mean_pooling = torch.mean(out, 1)
#         max_pooling, _ = torch.max(out, 1)
#         embed = torch.cat((mean_pooling, max_pooling), 1)
        embed=self.pooler(outputs[0])
        y_pred = self.classifier(self.dropout(embed))
        y_pred_target = torch.sigmoid(self.target_classifier(self.target_dropout(embed)))
        
        loss_token = None
        loss_target= None
        loss_label = None
        loss_total = None
        
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            ### Adding weighted
            
            # Only keep active parts of the loss
            if mask is not None:
                class_weights=torch.tensor([1.0,1.0],dtype=torch.float).to(input_ids.device)
                loss_funct = nn.CrossEntropyLoss(class_weights)
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                )
                loss_token = loss_funct(active_logits, active_labels)
            else:
                loss_token = loss_funct(logits.view(-1, 2), attn.view(-1))
            
            loss_total=self.impact_factor*loss_token
            
        if targets is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(y_pred_target.view(-1, self.num_targets), targets.view(-1, self.num_targets))
            loss_targets= loss_logits
            loss_total+=self.target_factor*loss_targets
            
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if(loss_total is not None):
                loss_total+=loss_label
            else:
                loss_total=loss_label
        if(loss_total is not None):
            return y_pred,y_pred_target,logits, loss_total
        else:
            return y_pred,y_pred_target,logits
        
        

        
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    