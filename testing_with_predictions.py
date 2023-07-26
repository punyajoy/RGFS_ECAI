import numpy as np
from Data_code.data import *
from Data_code.load_data import *
from Model_code.models import *
from Model_code.utils import *
from Eval_code.eval_scripts import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
import neptune.new as neptune
import GPUtil
import numpy as np
from datasets import list_datasets, load_dataset
# from apiconfig import *
import pandas as pd
from tqdm import tqdm
import argparse
import json
import time
from Data_code.utils import returnMask_test
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm_notebook


text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    segmenter="twitter", 
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

classes_in_dataset = {
    'Davidson': 3,
    'Founta': 3,
    'Basile': 2,
    'Olid': 2,
    'Waseem':2
}

datasets_labels_map = {
    "Founta": {
        "normal": 0,
        "hateful": 1,
        "abusive": 2
    },
    "Olid": {
        "NOT": 0,
        "OFF": 1
    },
    "Davidson": {
        "normal": 0,
        "hateful": 1,
        "offensive": 2
    },
    "Basile": {
        "normal": 0,
        "hateful": 1
    },
    "Waseem":{
        "normal": 0,
        "sexism": 1
    }
}

params={
 'dataset': None,
 'training_points': None, # CHANGE
 'model_path': 'bert-base-uncased',        # CHANGE
 'model_type': None,      # CHANGE
 'training_type': 'normal',
 'logging': 'local',
 'data_seed': None, # CHANGE
 'cache_path':'/home/punyajoy/HULK/Saved_models/',
 'learning_rate':1e-5,
 'device':'cuda',
 'num_classes': -1,
 'epochs': 20, # CHANGE
 'batch_size':4,
 'rationale_impact': 10,
 'attn_lambda':1,
 'train_rationale':False,
 'train_att':False,
 'save_model': True,
 'save_path':'Saved_Models/Domain_Adapt_New/DA/',
 'predictions_save_path': 'Model_Predictions/Final/',
 'random_rationales': False,
}


class modelPred_lime():
    def __init__(self, model_path = os.path.join('Saved_Models/Domain_Adapt_New/DA/', 'Olid', '50', '2021', 'Transform_Rationale_CrossAttn_CLS_Drpt_corrected' 'model_weights.pt')):
        self.device = torch.device("cuda")
        self.model_path = model_path

        rationale_predictor_model = Model_Rational_Label.from_pretrained("Saved_Models//Best_Toxic_BERT/BERT_toxic_rationale_2", params={'num_classes':2, 'rationale_impact':10,'target_impact':0,'targets_num':22},output_attentions = True,output_hidden_states = False).to(params['device'])
        self.rationale_predictor = modelPred(params=params, model=rationale_predictor_model)
        print("Loading pretrained model from " + str(self.model_path) + "...")
        pretrained_dict = torch.load(self.model_path)
        config = BertConfig()
        if('CrossAttn' in model_path):
            model = Transform_Rationale_CrossAttn_CLS_Drpt_corrected(config=config, params=params).to(params['device'])
        elif('SelfAttn' in model_path):
            model = Transform_Rationale_SelfAttn_Drpt_corrected(config=config, params=params).to(params['device'])
        else:
            config.num_labels=params['num_classes']
            model = BertForSequenceClassification(config=config).to(params['device'])

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                            (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        self.model = model
        print("Model Loaded!")

        self.model.cuda()  
        self.model.eval()
    
    # def process_path(self, model_path):
    #     model_name = model_path.split('/')[3]
    #     model_type = model_name.split('_')[2]
    #     return model_type
        
        
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    def tokenize(self, sentences, padding = True, max_len = 128):
        input_ids, attention_masks, token_type_ids, rationales = [], [], [], []
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast = False)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
        rationales = list(self.rationale_predictor.return_probab(sentences)[1])

        return {'input_ids': input_ids, 'attention_masks': attention_masks, 'rationales': torch.Tensor(rationales)}
    
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

        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], inputs['rationales'])
        sampler = SequentialSampler(data)
        
        return DataLoader(data, sampler=sampler, batch_size=2)
    
    def return_probab(self,sentences_list):
        """Input: should be a list of sentences"""
        """Output: probablity values"""
        device = self.device

        test_dataloader=self.process_data(sentences_list)
        logits_all=[]
        rationales_all=[]
        tokens_all=[]
        # Evaluate data 
        for step,batch in enumerate(test_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_rationales = batch[2].to(device)
            
            if('bert' in self.model_path):
                outputs = self.model(b_input_ids, b_input_mask)
            else:
                outputs = self.model(b_input_ids, b_input_mask, b_rationales)
        
            if type(self.model) == BertForSequenceClassification:
                logits = outputs.logits
            elif type(outputs) == tuple:
                logits = outputs[0]
            else:
                logits = outputs
            
            rationales_softmaxed = masked_softmax(b_rationales[:, :, 1], b_input_mask, dim=1).view(-1, 128, 1)
            
            logits = logits.detach().cpu().numpy()
            rationales_softmaxed=rationales_softmaxed.detach().cpu().numpy()
            tokens=b_input_ids.detach().cpu().numpy()
            logits_all+=list(logits)
            rationales_all+=list(rationales_softmaxed)
            tokens_all+=list(tokens)
        logits_all_final=[]
        for logits in logits_all:
            logits_all_final.append(list(softmax(logits)))
        # if self.flip:
        #     logits_array = np.array(logits_all_final)
        #     logits_array[:,[0, 1]] = logits_array[:,[1, 0]]
        #     return logits_array
        return np.array(logits_all_final),np.array(rationales_all),np.array(tokens_all)
        
def get_training_data(dict_map,model_path):
    final_output = []
    for key in tqdm(dict_map.keys()):
        annotation=dict_map[key]['label']
        text=dict_map[key]['text']
        post_id=key
        # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens_all,attention_masks=returnMask_test(dict_map[key], tokenizer)
        
        
        
        final_output.append([post_id, annotation, tokens_all, attention_masks])
    return final_output


def convert_data(test_data,params,list_dict,rational_present=True,topk=2):
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- how many words to select"""
    
    temp_dict={}
    for ele in list_dict:
        temp_dict[ele['annotation_id']]=ele['rationales'][0]['soft_rationale_predictions']
    
    test_data_modified=[]
    
    for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
        try:
            attention=temp_dict[row['id']]
        except KeyError:
            continue
        
        topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        new_text =[]
        new_attention =[]
        if(rational_present):
            new_attention =[0]
            new_text = [101]
            for i in range(len(row['tokens'])):
                if(i in topk_indices):
                    new_text.append(row['tokens'][i])
                    new_attention.append(row['attention'][0][i])
            new_attention.append(0)
            new_text.append(102)
        else:
            for i in range(len(row['tokens'])):
                if(i not in topk_indices):
                    new_text.append(row['tokens'][i])
                    new_attention.append(row['attention'][0][i])
        test_data_modified.append([row['id'],row['label'],new_text,new_attention])

    df=pd.DataFrame(test_data_modified,columns=test_data.columns)
    return df



def transform_dummy_data(sentences):
    post_id_list=['temp']*len(sentences)
    pred_list=['normal']*len(sentences)
    explanation_list=[]

def transform_attention(exp_list):
    max_attn=np.max(exp_list)
    min_attn=np.min(exp_list)
    diff=max_attn-min_attn
    exp_list_new=[]
    for exp in exp_list:
        exp_list_new.append((exp-min_attn)/diff)
    
    return exp_list_new

def standaloneEval(model_to_use, dataset_name, test_data=None, topk=2, rational=False):
    reverse_dict={datasets_labels_map[dataset_name][key]:key for key in datasets_labels_map[dataset_name].keys()}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    modelClass=modelPred_lime(model_to_use)
    
    print(test_data.head(5))
    
    sentences_all=[]
    for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
        sentences_all.append(row['text'])
    
    all_probab,all_rationales,all_tokens=modelClass.return_probab(sentences_all)

    list_dict=[]
    for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
        temp={}
        pred_id=np.argmax(all_probab[index])
        attention = transform_attention(all_rationales[index])
        tokens=all_tokens[index]
        pred_label=reverse_dict[pred_id]
        ground_label=row['label']
        temp["annotation_id"]=0
        temp["classification"]=pred_label
        
        
        temp_preds={}
        for key in reverse_dict.keys():
            temp_preds[reverse_dict[key]]=all_probab[index][key]
        temp["classification_scores"]=temp_preds
        temp['tokens']=tokens
        temp['attentions']=transform_attention(attention)
        list_dict.append(temp)

        

       
    del modelClass
    torch.cuda.empty_cache()
    return list_dict,test_data

# def standaloneEval(model_to_use, dataset_name, test_data=None, topk=2, rational=False):
#     reverse_dict={datasets_labels_map[dataset_name][key]:key for key in datasets_labels_map[dataset_name].keys()}
    
# #     if classes_in_dataset[dataset_name] == 2:
# #         reverse_dict={0:'normal',1:'abusive'}
# #     elif classes_in_dataset[dataset_name] == 3:
# #         reverse_dict={0:'normal',1:'hateful', 2: 'abusive'}
        
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
#     list_dict=[]
#     modelClass=modelPred_lime(model_to_use)
    
#     sentences_all=[]
#     for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
#         sentences_all.append(row['text'])
    
#     all_probab=modelClass.return_probab(sentences_all)
#     for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
#         temp={}
#         pred_id=np.argmax(all_probab[index])
#         pred_label=reverse_dict[pred_id]
#         ground_label=row['label']
#         temp["annotation_id"]=0
#         temp["classification"]=pred_label
#         temp_preds={}
#         for key in reverse_dict.keys():
#             temp_preds[reverse_dict[key]]=all_probab[index][key]
#         temp["classification_scores"]=temp_preds
#         list_dict.append(temp)
    
#     del modelClass
#     torch.cuda.empty_cache()
#     return list_dict,test_data
    
    
    
        
def get_final_dict_with_lime(model_name, dataset_name, test_data, topk):
    list_dict_org,test_data=standaloneEval(model_name, dataset_name, test_data=test_data, topk=topk)
#     test_data_with_rational=convert_data(test_data,params,list_dict_org,rational_present=True,topk=topk)
#     list_dict_with_rational,_=standaloneEval(model_name,dataset_name,test_data=test_data_with_rational, topk=topk,rational=True)
#     test_data_without_rational=convert_data(test_data,params,list_dict_org,rational_present=False,topk=topk)
#     list_dict_without_rational,_=standaloneEval(model_name,dataset_name,test_data=test_data_without_rational, topk=topk,rational=True)
#     
#     for ele1,ele2,ele3 in zip(list_dict_org,list_dict_with_rational,list_dict_without_rational):
#         ele1['sufficiency_classification_scores']=ele2['classification_scores']
#         ele1['comprehensiveness_classification_scores']=ele3['classification_scores']
#         final_list_dict.append(ele1)
#     for ele1 in list_dict_org:
#         ### these are just dummy results
#         ele1['sufficiency_classification_scores']=ele1['classification_scores']
#         ele1['comprehensiveness_classification_scores']=ele1['classification_scores']
#         final_list_dict.append(ele1)

    return list_dict_org

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    
    



if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Run Lime on Manually Annotated Data')

    # Add the arguments
    my_parser.add_argument('-dataset_name',
                           metavar='--dataset_name',
                           type=str,
                           default = 'Basile',
                           help='Name of Dataset to test')
    
    my_parser.add_argument('-model_name',
                           metavar='--model_name',
                           type=str,
                           default ='BERT_toxic_attn_100',
                           help='model to load')
    
    my_parser.add_argument('-random_seed',
                           metavar='--random_seed',
                           type=int,
                           default=2021,
                           help='random seed used')
    
    my_parser.add_argument('-training_points',
                           metavar='--num_test_points',
                           type=int,
                           default=50,
                           help='number of testing examples')
    
    args = my_parser.parse_args()

    dataset=args.dataset_name
    params['num_classes'] = classes_in_dataset[args.dataset_name]
    dataset_path='Dataset/NEW/'+dataset+'/Test.csv'
    model_name=args.model_name
    training_points=args.training_points
    random_seed=args.random_seed
    model_path=os.path.join('Saved_Models/Domain_Adapt_New/DA/', args.dataset_name, str(args.training_points), str(args.random_seed), args.model_name, 'model_weights.pt') #+dataset+'/'+model_name+'_'+str(training_points)+'_'+str(random_seed)

    
#     with open(dataset_path,'r') as infile:
#         dict_map=json.load(infile)
    
    
#     final_output=get_training_data(dict_map,model_path)
#     test_data=pd.DataFrame(final_output,columns=['id','label','tokens','attention'])
    
    test_data=pd.read_csv(dataset_path)
    final_dict=get_final_dict_with_lime(model_path, args.dataset_name, test_data, topk=5)
    
#     try:
#         final_dict=get_final_dict_with_lime(model_path,test_data,topk=5)
#     except OSError:
#         exit()
    path_name_explanation='prediction_exp_dicts/'+dataset+'_'+model_name+'_'+str(random_seed)+'_'+str(training_points)+'.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in final_dict))

