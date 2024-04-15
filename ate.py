
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch

# print(torch.cuda.is_available())
# print(torch.backends.mps.is_built())

root_path = './'

# use_mps = True if torch.has_mps else False
use_mps = torch.backends.mps.is_built()
os.chdir(root_path)

import warnings

warnings.filterwarnings('ignore')
import pandas as pd

from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from instructions import InstructionsHandler

task_name = 'ate'
experiment_name = 'lapt2014_iabsa3'
model_checkpoint = './model/tk-instruct-base-def-pos'
print('Experiment Name: ', experiment_name)
model_out_path = './Models'
model_out_path = os.path.join(model_out_path, task_name, f"{model_checkpoint.replace('/', '')}-{experiment_name}")
print('Model output path: ', model_out_path)

# Load the data
id_train_file_path = 'Dataset/SemEval14/Train/Laptops_Train_merge.csv'
id_test_file_path = './Dataset/SemEval14/Test/Laptops_Test.csv'
id_tr_df = pd.read_csv(id_train_file_path)
id_te_df = pd.read_csv(id_test_file_path, encoding='gbk')

id_val_file_path = './Dataset/SemEval14/Validation/Laptops_val.csv'
id_val_df = pd.read_csv(id_val_file_path)

# Get the input text into the required format using Instructions
instruct_handler = InstructionsHandler()

# Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
instruct_handler.load_instruction_set1()

# Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
loader = DatasetLoader(train_df_id=id_tr_df, test_df_id=id_te_df, val_df_id=id_val_df)
if loader.train_df_id is not None:
    loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms',
                                                          instruct_handler.ate['bos_instruct1'],
                                                          instruct_handler.ate['eos_instruct'])
if loader.test_df_id is not None:
    loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms',
                                                         instruct_handler.ate['bos_instruct1'],
                                                         instruct_handler.ate['eos_instruct'])
if loader.val_df_id is not None:
    loader.val_df_id = loader.create_data_in_ate_format(loader.val_df_id, 'term', 'raw_text', 'aspectTerms',
                                                        instruct_handler.ate['bos_instruct1'],
                                                        instruct_handler.ate['eos_instruct'])
# Create T5 utils object
t5_exp = T5Generator(model_checkpoint)

# Tokenize Dataset
id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

# Training arguments
training_args = {
    'output_dir': model_out_path,
    'evaluation_strategy': "epoch",
    'learning_rate': 5e-5,
    'lr_scheduler_type': 'cosine',
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'num_train_epochs': 12,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'save_strategy': 'no',
    'load_best_model_at_end': False,
    'push_to_hub': False,
    'eval_accumulation_steps': 1,
    'predict_with_generate': True,
}
# Train model
model_trainer = t5_exp.train(id_tokenized_ds, **training_args)
## Inference
# Load the data
id_train_file_path = 'Dataset/SemEval14/Train/Laptops_Train_merge.csv'
id_test_file_path = 'Dataset/SemEval14/Test/Laptops_Test.csv'
id_tr_df = pd.read_csv(id_train_file_path)
id_te_df = pd.read_csv(id_test_file_path, encoding='gbk')

# ood_train_file_path = './Dataset/SemEval14/Train/Restaurants_Train.csv'
# ood_test_file_path = './Dataset/SemEval14/Test/Restaurants_Test.csv'
# ood_tr_df = pd.read_csv(ood_train_file_path)
# ood_te_df = pd.read_csv(ood_test_file_path)

# Get the input text into the required format using Instructions
instruct_handler = InstructionsHandler()

# Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
instruct_handler.load_instruction_set1()

# Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
loader = DatasetLoader(train_df_id=id_tr_df, test_df_id=id_te_df)  #, train_df_ood=ood_tr_df, test_df_ood=ood_te_df)
if loader.train_df_id is not None:
    loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms',
                                                          instruct_handler.ate['bos_instruct1'],
                                                          instruct_handler.ate['eos_instruct'])
if loader.test_df_id is not None:
    loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms',
                                                         instruct_handler.ate['bos_instruct1'],
                                                         instruct_handler.ate['eos_instruct'])

# if loader.train_df_ood is not None:
#     loader.train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct'])
# if loader.test_df_ood is not None:
#     loader.test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct'])
#%%
# Model inference - Loading from Checkpoint
t5_exp = T5Generator(model_out_path)

# Tokenize Datasets
id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

# Get prediction labels - Training set
id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='train',
                                      # trained_model_path=model_out_path,
                                      batch_size=16)
id_tr_labels = [i.strip() for i in id_ds['train']['labels']]

# Get prediction labels - Testing set
id_te_pred_labels = t5_exp.get_labels(tokenized_dataset=id_tokenized_ds, sample_set='test',
                                      # trained_model_path=model_out_path,
                                      batch_size=16)
id_te_labels = [i.strip() for i in id_ds['test']['labels']]
#%%
p, r, f1, _ = t5_exp.get_metrics(id_tr_labels, id_tr_pred_labels)
print('Train Precision: ', p)
print('Train Recall: ', r)
print('Train F1: ', f1)

p, r, f1, _ = t5_exp.get_metrics(id_te_labels, id_te_pred_labels)
print('Test Precision: ', p)
print('Test Recall: ', r)
print('Test F1: ', f1)