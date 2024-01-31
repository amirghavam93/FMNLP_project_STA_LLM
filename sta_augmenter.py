
import pandas as pd
import jieba
from tqdm import tqdm
import os
import pickle
from augmenter import TextAugmenter
from keywords_extractor import role_kws_extraction_single


dataset = "newsgroups"
strategy = "global"
methods = ['de', 're', 'in', 'sw', 're']
p = 0.1
bar = "Q2"
n_aug = 1
best_practice = False
ablation_without = None

dataset_name = 'newsgroups'
output_dir = f"data/{dataset_name}/selective_{strategy}_{'_'.join(methods)}_{p}_{bar}_{n_aug}/"
if ablation_without is not None:
    output_dir = f"data/ablation_study/{dataset_name}/selective_{strategy}_{'_'.join(methods)}_{p}_{bar}_{n_aug}/"
if os.path.exists(output_dir) == False:
    os.makedirs(output_dir)


train_path = f'data/{dataset_name}/{dataset_name}_1000.csv'
global_lr_dict_path = f'saved_keywords/global_lr_dict_{dataset_name}.pkl'
global_ls_dict_path = f'saved_keywords/global_ls_dict_{dataset_name}.pkl'
global_kws_dict_path = f'saved_keywords/global_kws_dict_{dataset_name}.pkl'

assert os.path.exists(global_lr_dict_path) and os.path.exists(global_ls_dict_path), "file not exists!"
assert os.path.exists(global_kws_dict_path), 'file not exists!'

raw_train_df = pd.read_csv(train_path)
raw_train_df = raw_train_df.dropna()
raw_train_df = raw_train_df[raw_train_df.text != '']


texts = list(raw_train_df['text'])
labels = list(raw_train_df['label'])

with open(global_lr_dict_path, 'rb') as f:
    global_lr_dict = pickle.load(f)
with open(global_ls_dict_path, 'rb') as f:
    global_ls_dict = pickle.load(f)
with open(global_kws_dict_path, 'rb') as f:
    global_kws_dict = pickle.load(f)

TA = TextAugmenter('en')

puncs = ',.，。!?！？;；、'
punc_list = [w for w in puncs]
special_tokens = ",./;\`~<>?:\"，。/；‘’“”、｜《》？～· \n[]{}【】「」（）()0123456789０１２３４５６７８９" \
            "，．''／；\｀～＜＞？：＂,。／;‘’“”、|《》?~·　\ｎ［］｛｝【】「」("")（） "
stop_words = TA.stop_words
skip_words = [t for t in special_tokens] + stop_words

mix_contents = []
mix_labels = []
mix_contents += texts
mix_labels += labels

print_info = False
for method in methods:
    aug_filename = output_dir+f'train_{method}.csv'
    augmented_texts = []
    for i in range(n_aug): # augment multiple times
        for text,label in zip(tqdm(texts), labels):

            label = str(label)
            words = TA.tokenizer(text)
            print(words)
            if strategy == 'local':
                kws = role_kws_extraction_single(words, label, global_ls_dict, global_lr_dict, bar=bar, skip_words=skip_words)
            elif strategy == 'global':
                kws = global_kws_dict[label]
                
            if ablation_without == 'lr': 
                print(f'>>> ABLATION STUDY: WITHOUT [lr]')
                kws['ccw'] = kws['ccw'] + kws['scw']
                kws['iw'] = kws['iw'] + kws['fcw']
                kws['fcw'] = []
                kws['scw'] = []
            if ablation_without == 'ls':
                print(f'>>> ABLATION STUDY: WITHOUT [ls]')
                kws['ccw'] = kws['ccw'] + kws['fcw']
                kws['iw'] = kws['iw'] + kws['scw']
                kws['fcw'] = []
                kws['scw'] = []
                

            if method == 'de':
                new_words = TA.aug_by_deletion(text, p, 'selective', print_info=print_info,
                                               selected_words=kws['scw']+kws['fcw']+kws['iw'])  # except ccw
            elif method == 're':
                new_words = TA.aug_by_replacement(text, p, 'selective', print_info=print_info,
                                                  selected_words=kws['scw']+kws['fcw']+kws['iw'])  # except ccw
            elif method == 'in':
                new_words = TA.aug_by_insertion(text, p, 'selective', print_info=print_info,
                                                selected_words=kws['ccw']+kws['scw']+kws['iw'])  # except fcw
            elif method == 'sw':
                new_words = TA.aug_by_swap(text, p, 'selective', print_info=print_info,
                                           selected_words=kws['iw'])  # iw better
            elif method == 'se':
                new_words = TA.aug_by_selection(text, print_info=print_info,
                                                selected_words=kws['ccw']+kws['iw']+punc_list)
            else:
                raise NotImplementedError()

            joint_str = ' '
            new_text = joint_str.join(new_words)
            for punc in puncs: 
                new_text = new_text.replace(' '+punc, punc)
            augmented_texts.append(new_text)
    new_df = pd.DataFrame({'text': texts+augmented_texts, 'label': labels*(n_aug+1)})
    # new_df = pd.DataFrame({'text': texts+augmented_texts, 'label': labels*(args.n_aug+1)})

    new_df.to_csv(aug_filename)
    print('saved to %s'%aug_filename)
    mix_contents += augmented_texts
    mix_labels += labels*n_aug

mix_filename = output_dir+'train_mix.csv'
mix_df = pd.DataFrame({'text': mix_contents, 'label': mix_labels})
mix_df.to_csv(mix_filename)
print('saved to %s'%mix_filename)

print(f'>>> before augmentation: {len(texts)}')
print(f'>>> after augmentation: {len(mix_contents)}')

