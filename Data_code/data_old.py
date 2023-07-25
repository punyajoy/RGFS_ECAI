from .utils import *
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import re

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


# For DA
class Hatexplain_Dataset_new():
    def __init__(self, data, rationale_predictor, params=None, tokenizer=None, train = False):
        # print(params['cache_path'])
        
        self.rationale_predictor = rationale_predictor
        self.data = data
        self.batch_size = 16
        self.train = train
        self.params= params
        
        if params['num_classes'] == 3:
            self.label_dict = {0: 0,
                                1: 1,
                                2: 2}
        elif params['num_classes'] == 2:
            self.label_dict = {0: 1,
                                1: 0,
                                2: 1}
                                    
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.labels, self.attn = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels)
        
    def tokenize(self, sentences, padding = True, max_len = 128):
        input_ids, attention_masks, token_type_ids = [], [], []
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
    
    def process_masks(self, masks):
        mask = []
        for idx in range(len(masks[0])):
            votes = 0
            for at_mask in masks:
                if at_mask[idx] == 1: votes+=1
            if votes > len(masks)/2: mask.append(1)
            else: mask.append(0)
        return mask
    
    def process_mask_attn(self,masks,label):
        if(label=='non_toxic'):
            at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
        else:
            at_mask_fin=masks
            at_mask_fin=np.mean(at_mask_fin,axis=0)
            at_mask_fin=softmax(at_mask_fin)
        return list(at_mask_fin)
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        print(len(data))
        for row in data:
            label = max(set(row['annotators']['label']), key = row['annotators']['label'].count)
            sentence = ' '.join(row['post_tokens'])
            sentences.append(sentence)
            labels.append(self.label_dict[label])
        inputs = self.tokenize(sentences)
        attn = list(rationale_predictor.return_probab(sentences)[1])
        return inputs, torch.Tensor(labels), torch.Tensor(attn)
    
    def get_dataloader(self, inputs, attn, labels, train = False):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], attn, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

    

# For DA
class Normal_Dataset_new(Hatexplain_Dataset_new):
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    def dummy_attention(self,inputs):
        attn=[]
        for sent in inputs['input_ids']:
            temp=[0]*len(sent)
            attn.append(temp)
        return attn
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        print(len(data))
        for label, sentence in zip(list(data['label']), list(data['text'])):
            #label = self.label_dict[label]
            
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
            labels.append(label)
        inputs = self.tokenize(sentences)
#         attn = self.dummy_attention(inputs)
        attn = list(self.rationale_predictor.return_probab(sentences)[1])
        return inputs, torch.Tensor(labels), torch.Tensor(attn)

    
    

    
    
class Hatexplain_Dataset():
    def __init__(self, data, params=None, tokenizer=None, train = False):
        print(params['cache_path'])
        self.data = data
        self.batch_size = params['batch_size']
        self.train = train
        self.params= params
        
        if params['num_classes'] == 3:
            self.label_dict = {0: 0,
                                1: 1,
                                2: 2}
        elif params['num_classes'] == 2:
            self.label_dict = {0: 1,
                                1: 0,
                                2: 1}
                                    
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.labels, self.attn = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels)
        
    def tokenize(self, sentences, padding = True, max_len = 128):
        input_ids, attention_masks, token_type_ids = [], [], []
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
    
    def process_masks(self, masks):
        mask = []
        for idx in range(len(masks[0])):
            votes = 0
            for at_mask in masks:
                if at_mask[idx] == 1: votes+=1
            if votes > len(masks)/2: mask.append(1)
            else: mask.append(0)
        return mask
    
    def process_mask_attn(self,masks,label):
        if(label=='non_toxic'):
            at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
        else:
            at_mask_fin=masks
            at_mask_fin=np.mean(at_mask_fin,axis=0)
            at_mask_fin=softmax(at_mask_fin)
        return list(at_mask_fin)
    
    
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        print(len(data))
        for row in data:
            word_tokens_all, word_mask_all = returnMask(row, self.tokenizer)
            label = max(set(row['annotators']['label']), key = row['annotators']['label'].count)
            if(self.params['train_att']):
                at_mask = self.process_mask_attn(word_mask_all,label)
            elif(self.params['train_rationale']):
                at_mask = self.process_masks(word_mask_all)
            else:
                at_mask = []
            sentence = ' '.join(row['post_tokens'])
            sentences.append(sentence)
            labels.append(self.label_dict[label])
            at_mask = at_mask + [0]*(128-len(at_mask))
            attn.append(at_mask)
        inputs = self.tokenize(sentences)
        return inputs, torch.Tensor(labels), torch.Tensor(attn)
    
    def get_dataloader(self, inputs, attn, labels, train = True):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], attn, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=True)
    
    
    


    
    
class Normal_Dataset(Hatexplain_Dataset):
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    def dummy_attention(self,inputs):
        attn=[]
        for sent in inputs['input_ids']:
            temp=[0]*len(sent)
            attn.append(temp)
        return attn
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        print(len(data))
        for label, sentence in zip(list(data['label']), list(data['text'])):
            #label = self.label_dict[label]
            
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
            labels.append(label)
        inputs = self.tokenize(sentences)
        attn = self.dummy_attention(inputs)
        return inputs, torch.Tensor(labels), torch.Tensor(attn)