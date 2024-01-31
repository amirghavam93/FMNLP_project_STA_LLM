from gensim.models.keyedvectors import KeyedVectors
import math
import numpy as np
import nltk
import pickle
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# Compute tf, idf values
def get_tf(t, document_words):
    """
    Compute term frequency in a document
    :param t: term
    :param document_words: list of words in the document
    :return: term frequency
    """
    return document_words.count(t) / len(document_words)


def get_idf(t, ds):
    """
    Compute inverse document frequency in a dataset
    :param t: term
    :param ds: dataset (collection of documents)
    :return: inverse document frequency
    """
    n = len(ds)
    df = len([d for d in ds if t in d])
    return math.log(n / (1 + df))


def get_wllr(in_class_freq, out_class_freq):
    """
    Weighted Log Likelihood Ratio (WLLR) for term classification relevance.
    :param in_class_freq: frequency of term in class
    :param out_class_freq: frequency of term outside the class
    :return: WLLR score
    """
    wllr = in_class_freq * math.log10(in_class_freq / out_class_freq)
    return wllr

def get_median(scores):
    scores = sorted(scores, reverse=True)
    l = len(scores)
    if l % 2 == 0:
        return (scores[int(l/2-1)]+scores[int(l/2)])/2
    else:
        return scores[int((l-1)/2)]

def get_quartiles(scores):
    """
    Calculate quartiles for a list of scores.
    :param scores: List of numeric scores
    :return: Dictionary with Q1, Q2 (median), and Q3 quartiles
    """
    try:
        scores = sorted(scores, reverse=True)
        if len(scores) == 1:
            return {'Q1':scores[0],'Q2':scores[0],'Q3':scores[0]}
        if len(scores) == 2:
            return {'Q1':(scores[0]+(scores[0]+scores[1])/2)/2, 'Q2':(scores[0]+scores[1])/2, 'Q3':(scores[1]+(scores[0]+scores[1])/2)/2}

        l = len(scores)
        if l % 2 == 0:
            return {'Q1':get_median(scores[:int(l/2)]), 'Q2': get_median(scores), 'Q3': get_median(scores[int(l/2):])}
        else:
            return {'Q1':get_median(scores[:int((l-1)/2)]), 'Q2': get_median(scores), 'Q3': get_median(scores[int((l+1)/2):])}
    except Exception as e:
        print(e)
        print('scores',scores)

def normalize(score, Min, Max): # min-max normalization
    if score == Min:
        return 1e-5
    return (score-Min)/(Max-Min)


class KeywordsExtractor:
    def __init__(self, lang='en'):
        assert lang == 'en', "Only 'en' (English) language is supported"
        print('Language: English')
        self.lang = lang
        self.stop_words = []

        # Load English word vectors (Google News vectors)
        print('Loading word vectors......')
        self.w2v_model = KeyedVectors.load_word2vec_format("weights/GoogleNews-vectors-negative300.bin", binary=True, unicode_errors='ignore')

        # Load English stopwords
        with open('stopwords/en_stopwords.txt', encoding='utf8') as f:
            self.stop_words = f.read().splitlines()

        self.tokenizer = nltk.tokenize.word_tokenize
        self.vocab = self.w2v_model.index_to_key

    def get_text_vec(self, s):
        """
        Get the word2vec representation of a given string.
        :param s: string (word, phrase, sentence)
        :return: vector representation of the string
        """
        if s in self.vocab:
            v = self.w2v_model[s]
        else:
            v = np.zeros_like(self.w2v_model['the'])  # Placeholder vector
            count = 0
            words = self.tokenizer(s)
            for t in words:
                if t in self.vocab:
                    v += self.w2v_model[t]
                    count += 1
            if count:
                v /= count
            else:
                v = self.w2v_model['the']  # Fallback to a default vector
        return v

    def compute_similarity_by_vector(self, v1, v2):
        """
        Compute cosine similarity between two vectors.
        :param v1: vector 1
        :param v2: vector 2
        :return: cosine similarity
        """
        return self.w2v_model.cosine_similarities(v1, v2.reshape(1, -1))[0]

    def compute_similarity_by_text(self, s1, s2):
        """
        Compute cosine similarity between two text strings.
        :param s1: string 1
        :param s2: string 2
        :return: cosine similarity
        """
        vec_pair = [self.get_text_vec(s1), self.get_text_vec(s2)]
        return self.w2v_model.cosine_similarities(vec_pair[0], vec_pair[1].reshape(1, -1))[0]

    def compute_label_similarity(self, contents, labels, label_desc_dict=None, num_words=None):
        """
        Compute label similarity for each word in a class.
        :param contents: List of documents
        :param labels: Corresponding labels for the documents
        :param label_desc_dict: Optional dictionary of label descriptions
        :param num_words: Maximum number of words to consider in each document
        :return: Tuple of global label similarity dictionary and sorted label similarity dictionary
        """
        assert len(contents) == len(labels), 'Contents and labels must be of the same length.'
        global_ls_dict = {}
        sorted_ls_dict = {}
        words_dict = {label: [] for label in set(labels)}

        # Tokenizing and grouping words by label
        for content, label in zip(tqdm(contents), labels):
            if num_words:
                words_dict[label] += self.tokenizer(content)[:num_words]
            else:
                words_dict[label] += self.tokenizer(content)

        # Computing label similarity
        label_desc_vec = {}
        for label in set(labels):
            global_ls_dict[label] = {}
            for w in tqdm(list(set(words_dict[label]))):
                if label_desc_dict is None:
                    global_ls_dict[label][w] = self.compute_similarity_by_text(w, label)
                else:
                    # Compute label description vector once to optimize performance
                    if label in label_desc_vec:
                        label_v = label_desc_vec[label]
                    else:
                        label_v = self.get_text_vec(label_desc_dict[label])
                        label_desc_vec[label] = label_v
                    word_v = self.get_text_vec(w)
                    global_ls_dict[label][w] = self.compute_similarity_by_vector(word_v, label_v)

            # Sorting by similarity
            sorted_ls_dict[label] = [(pair[0], pair[1]) for pair in sorted(global_ls_dict[label].items(),
                                                                           key=lambda kv: kv[1], reverse=True)]
        return global_ls_dict, sorted_ls_dict

    def compute_label_correlation(self, contents, labels, num_words=None):
        """
        Compute label correlation for each word in a class.
        :param contents: List of documents
        :param labels: Corresponding labels for the documents
        :param num_words: Maximum number of words to consider in each document
        :return: Tuple of global label correlation dictionary and sorted label correlation dictionary
        """
        assert len(contents) == len(labels), 'Contents and labels must be of the same length.'
        global_doc_count = {}
        global_wllr_dict = {}
        sorted_wllr_dict = {}

        # Counting documents for each word in each label
        for label in set(labels):
            global_doc_count[label] = {'TOTAL_COUNT': 0}
            global_wllr_dict[label] = {}
        for content, label in zip(tqdm(contents), labels):
            words = self.tokenizer(content)[:num_words] if num_words else self.tokenizer(content)
            global_doc_count[label]['TOTAL_COUNT'] += 1
            for w in set(words):
                global_doc_count[label][w] = global_doc_count[label].get(w, 0) + 1

        # Computing label correlation
        for label in set(labels):
            num_in_class_docs = global_doc_count[label]['TOTAL_COUNT']
            num_out_class_docs = sum([global_doc_count[l]['TOTAL_COUNT'] for l in set(labels) if l != label])
            for w in global_doc_count[label]:
                if w == 'TOTAL_COUNT':
                    continue
                in_count = global_doc_count[label][w]
                out_count = max(sum([global_doc_count[l].get(w, 0) for l in set(labels) if l != label]), 1e-5)
                in_class_freq = in_count / num_in_class_docs
                out_class_freq = out_count / num_out_class_docs
                global_wllr_dict[label][w] = get_wllr(in_class_freq, out_class_freq)

            # Sorting by WLLR
            sorted_wllr_dict[label] = [(pair[0], pair[1]) for pair in sorted(global_wllr_dict[label].items(),
                                                                             key=lambda kv: kv[1], reverse=True)]
        return global_wllr_dict, sorted_wllr_dict

    def extract_global_role_kws(self, labels, sorted_ls_dict, sorted_wllr_dict):
        """
        Extract global class-indicating words based on high label similarity and correlation.
        :param labels: List of unique labels
        :param sorted_ls_dict: Sorted label similarity dictionary
        :param sorted_wllr_dict: Sorted label correlation (WLLR) dictionary
        :return: Dictionary of global class-indicating words
        """
        global_class_indicating_words = {}
        global_fake_class_indicating_words = {}
        for label in set(labels):
            # Words with high label correlation
            high_lr = [w[0] for w in sorted_wllr_dict[label][:int(0.2 * len(sorted_wllr_dict[label]))]]
            # Words with high label similarity
            high_ls = [w[0] for w in sorted_ls_dict[label][:int(0.3 * len(sorted_ls_dict[label]))]]
            global_class_indicating_words[label] = [w for w in high_lr if w in high_ls]
            # Words with low label similarity but high label correlation (potential noise)
            low_ls = [w[0] for w in sorted_ls_dict[label][int(0.8 * len(sorted_ls_dict[label])):]]
            global_fake_class_indicating_words[label] = [w for w in high_lr if w in low_ls]
        return global_class_indicating_words, global_fake_class_indicating_words

    def global_role_kws_extraction_one_line(self, contents, labels, label_desc_dict=None, num_words=None, output_dir='.', name='', overwrite=False):
        """
        A streamlined function for extracting global role keywords in one go.
        :param contents: List of documents
        :param labels: Corresponding labels for the documents
        :param label_desc_dict: Optional dictionary of label descriptions
        :param num_words: Maximum number of words to consider in each document
        :param output_dir: Directory for saving results
        :param name: Name for the saved file
        :param overwrite: Flag to overwrite existing files
        :return: Dictionary containing global label similarity, correlation, and roles
        """
        # Paths for saving data
        ls_save_path = f'{output_dir}/global_ls_dict_{name}.pkl'
        lr_save_path = f'{output_dir}/global_lr_dict_{name}.pkl'
        global_roles_save_path = f'{output_dir}/global_kws_dict_{name}.pkl'
        
        # Compute and save label similarity
        if os.path.exists(ls_save_path) and not overwrite:
            with open(ls_save_path, 'rb') as f:
                global_ls_dict = pickle.load(f)
        else:
            global_ls_dict, sorted_ls_dict = self.compute_label_similarity(contents, labels, label_desc_dict, num_words)
            with open(ls_save_path, 'wb') as f:
                pickle.dump(global_ls_dict, f)

        # Compute and save label correlation
        if os.path.exists(lr_save_path) and not overwrite:
            with open(lr_save_path, 'rb') as f:
                global_lr_dict = pickle.load(f)
        else:
            global_lr_dict, sorted_lr_dict = self.compute_label_correlation(contents, labels, num_words)
            with open(lr_save_path, 'wb') as f:
                pickle.dump(global_lr_dict, f)

        # Extract and save global role keywords
        if os.path.exists(global_roles_save_path) and not overwrite:
            print('global roles dict already exists at: ',global_roles_save_path)
            with open(global_roles_save_path, 'rb') as f:
                global_roles_dict = pickle.load(f)
        else:
            global_roles_dict = global_role_kws_extraction(global_lr_dict, global_ls_dict, list(set(labels)))
            print('First level keys: ',list(global_roles_dict.keys()))
            print('Second level keys: ',list(global_roles_dict[list(global_roles_dict.keys())[0]].keys()))
            with open(f'{output_dir}/global_kws_dict_{name}.pkl','wb') as f:
                pickle.dump(global_roles_dict, f)
                print('already saved at',f'{output_dir}/global_kws_dict_{name}.pkl')

        return {
            'global_ls': global_ls_dict,
            'global_lr': global_lr_dict,
            'global_roles': global_roles_dict
        }

    # Additional helper methods can be added here
    def preprocess_text(self, text):
        """
        Preprocess the text by lowercasing, removing stopwords, and any other required processing.
        :param text: The text to preprocess
        :return: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = self.tokenizer(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def refine_keywords(self, keywords, top_n=10):
        """
        Refine the extracted keywords, selecting the top_n keywords based on some criteria.
        :param keywords: List of extracted keywords
        :param top_n: Number of top keywords to select
        :return: List of refined top_n keywords
        """
        # Example refinement process, can be customized
        return keywords[:top_n]

def global_role_kws_extraction(global_lr_dict, global_ls_dict, labels):
    puncs = ",./;\`~<>?:\"，。/；‘’“”、｜《》？～· \n[]{}【】「」（）()0123456789０１２３４５６７８９" \
            "，．''／；\｀～＜＞？：＂,。／;‘’“”、|《》?~·　\ｎ［］｛｝【】「」("")（） "
    kws_dict = {}
    for label in labels:
        lr_dict, ls_dict = global_lr_dict[label], global_ls_dict[label]
        lr_bar = get_quartiles(list(lr_dict.values()))
        ls_bar = get_quartiles(list(ls_dict.values()))
        lr_min, lr_max = min(list(lr_dict.values())), max(list(lr_dict.values()))
        ls_min, ls_max = min(list(ls_dict.values())), max(list(ls_dict.values()))

        words_lr_sorted = [p[0] for p in sorted(lr_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        words_ls_sorted = [p[0] for p in sorted(ls_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

        ccw_dict, scw_dict, fcw_dict, iw_dict = {}, {}, {}, {}
        for w in ls_dict: 
            if lr_dict[w] >= lr_bar['Q1'] and ls_dict[w] >= ls_bar['Q1']:  # ccw
                ccw_dict[w] = normalize(lr_dict[w], lr_min, lr_max) * normalize(ls_dict[w], ls_min, ls_max)
            if lr_dict[w] < lr_bar['Q3'] and ls_dict[w] >= ls_bar['Q1']:  # scw
                scw_dict[w] = normalize(ls_dict[w], ls_min, ls_max) - normalize(lr_dict[w], lr_min, lr_max)
            if lr_dict[w] >= lr_bar['Q1'] and ls_dict[w] < ls_bar['Q3']:  # fcw
                fcw_dict[w] = normalize(lr_dict[w], lr_min, lr_max) - normalize(ls_dict[w], ls_min, ls_max)
            if lr_dict[w] < lr_bar['Q3'] and ls_dict[w] < ls_bar['Q3']:  # iw
                iw_dict[w] = 1 / (normalize(lr_dict[w], lr_min, lr_max) * normalize(ls_dict[w], ls_min, ls_max))
        ccw_sorted = [p[0] for p in sorted(ccw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        scw_sorted = [p[0] for p in sorted(scw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        fcw_sorted = [p[0] for p in sorted(fcw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        iw_sorted = [p[0] for p in sorted(iw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        kws_dict[label] = {'lr': words_lr_sorted,
                            'ls': words_ls_sorted,
                            'ccw': ccw_sorted,
                            'scw': scw_sorted,
                            'fcw': fcw_sorted,
                            'iw': iw_sorted}
    return kws_dict




def role_kws_extraction_single(words, label, global_ls_dict, global_lr_dict, bar='Q2',skip_words=[]):
    lr_dict = {}
    ls_dict = {}
    
    for w in set(words):
        # filter punctuations and stop words
        if w in skip_words:
            continue

        print(global_ls_dict[label])
        ls_dict[w] = global_ls_dict[label][w]
        # ls_dict[w] = global_ls_dict['0'][label][w]

        lr_dict[w] = global_lr_dict[label][w]
        # lr_dict[w] = global_lr_dict['0'][label][w]

    if len(ls_dict) == 0: 
        for w in set(words):
            ls_dict[w] = global_ls_dict[label][w] 
            lr_dict[w] = global_lr_dict[label][w]
    lr_bar = get_quartiles(list(lr_dict.values()))[bar]
    ls_bar = get_quartiles(list(ls_dict.values()))[bar]
    lr_min, lr_max = min(list(lr_dict.values())), max(list(lr_dict.values()))
    ls_min, ls_max = min(list(ls_dict.values())), max(list(ls_dict.values()))
    words_lr_sorted = [p[0] for p in sorted(lr_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    words_ls_sorted = [p[0] for p in sorted(ls_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

    ccw_dict, scw_dict, fcw_dict, iw_dict = {}, {}, {}, {}
    for w in ls_dict: 
        if lr_dict[w] >= lr_bar and ls_dict[w] >= ls_bar: # ccw
            ccw_dict[w] = normalize(lr_dict[w],lr_min,lr_max) * normalize(ls_dict[w],ls_min,ls_max)
        if lr_dict[w] < lr_bar and ls_dict[w] >= ls_bar: # scw
            scw_dict[w] = normalize(ls_dict[w],ls_min,ls_max) / normalize(lr_dict[w],lr_min,lr_max)
        if lr_dict[w] >= lr_bar and ls_dict[w] < ls_bar: # fcw
            fcw_dict[w] = normalize(lr_dict[w],lr_min,lr_max) / normalize(ls_dict[w],ls_min,ls_max)
        if lr_dict[w] < lr_bar and ls_dict[w] < ls_bar: # iw
            iw_dict[w] = 1 / (normalize(lr_dict[w],lr_min,lr_max) * normalize(ls_dict[w],ls_min,ls_max))
    ccw_sorted = [p[0] for p in sorted(ccw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    scw_sorted = [p[0] for p in sorted(scw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    fcw_sorted = [p[0] for p in sorted(fcw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    iw_sorted = [p[0] for p in sorted(iw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    kws_dict = {'lr': words_lr_sorted,
                'ls': words_ls_sorted,
                'ccw': ccw_sorted,
                'scw': scw_sorted,
                'fcw': fcw_sorted,
                'iw':iw_sorted}
    return kws_dict