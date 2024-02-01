import nltk
import random
from random import shuffle
import pickle
from logging import Logger
from nltk.corpus import wordnet

logger = Logger('text augmenter')
random.seed(1)

class TextAugmenter:
    def __init__(self, lang, using_wordnet=False):
        """
        Initializes the TextAugmenter instance.
        
        Parameters:
        - lang (str): Specifies the language for augmentation ('en' for English).
        - using_wordnet (bool): Flag to indicate whether to use WordNet for finding synonyms.
        """
        assert lang == 'en', "This version supports only 'en' for English"
        print(f'Language: English')
        self.lang = lang
        self.stop_words = []
        self.using_wordnet = using_wordnet

        # Load similar words dictionary for English
        with open('weights/en_similars_dict.pkl', 'rb') as f:
            self.similar_words_dict = pickle.load(f)

        # Load English stop words
        with open('stopwords/en_stopwords.txt', encoding='utf8') as f:
            self.stop_words = [line.strip() for line in f.readlines()]

        self.vocab = list(self.similar_words_dict.keys())
    
    def tokenizer(self, text):
        """
        Tokenizes the input text into words.

        Parameters:
        - text (str): The input string to be tokenized.
        """
        return nltk.tokenize.word_tokenize(text)
    
    def get_similar_words(self, word):
        """
        Retrieves similar words for a given word using a pre-loaded dictionary or WordNet.

        Parameters:
        - word (str): The word for which similar words are to be found.
        """
        if self.using_wordnet:
            print('Using WordNet!')
            word = word.lower()
            similars = set()
            for w in wordnet.synsets(word):
                for l in w.lemmas():
                    sim = l.name().replace("_", " ").replace("-", " ").lower()
                    sim = "".join([char for char in sim if char in ' qwertyuiopasdfghjklzxcvbnm'])
                    similars.add(sim)
            if word in similars:
                similars.remove(word)
            return list(similars)
        
        if word not in self.vocab:
            return []
        else:
            return self.similar_words_dict.get(word)

    def aug_by_replacement(self, text, p, mode='random', selected_words=[], print_info=False):
        """
        Augments text by replacing words with their synonyms.

        Parameters:
        - text (str): The input text to augment.
        - p (float): Proportion of words to be replaced.
        - mode (str): 'random' or 'selective'. In 'selective' mode, only words in 'selected_words' are replaced.
        - selected_words (list): List of words to be replaced in 'selective' mode.
        - print_info (bool): If True, prints information about the replacements.
        """
        # words = self.tokenizer(text)
        # n = max(1, int(p * len(words)))
        # new_words = words.copy()
        # replacement_res = []
        # for word in (selected_words if mode == 'selective' else words):
        #     if word not in self.stop_words and word in self.vocab:
        #         similars = self.get_similar_words(word)
        #         if similars:
        #             similar = random.choice(similars)
        #             new_words = [similar if w == word else w for w in new_words]
        #             replacement_res.append((word, similar))
        # if print_info:
        #     print('replacement info:', replacement_res)
        # return ' '.join(new_words)
        words = self.tokenizer(text)
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        n = max(1, int(p * len(words)))
        new_words = words.copy()
        if mode == 'selective' and len(selected_words) > 0:
            selected_words = list(set(selected_words).intersection(set(words)))

            random.shuffle(selected_words)
            replacement_word_list = selected_words[:] 
            if len(selected_words) < n: 
                try:
                    replacement_word_list += random.sample(
                        list([word for word in words if word not in self.stop_words+selected_words]),
                        n - len(selected_words))
                except:
                    pass 
        else:
            replacement_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(replacement_word_list)
        num_replaced = 0
        replacement_res = [] 
        for word in replacement_word_list:
            similars = self.get_similar_words(word)
            if len(similars) >= 1:
                similar = random.choice(similars)
                new_words = [similar if w == word else w for w in new_words]
                replacement_res.append((word, similar))
                num_replaced += 1

            if num_replaced >= n:
                break
        if print_info:
            print('replacement info:', replacement_res)
        return new_words

    def aug_by_insertion(self, text, p, mode='random', selected_words=[], print_info=False):
        """
        Augments text by inserting similar or given words at random positions.

        Parameters:
        - text (str): The input text to augment.
        - p (float): Proportion of words to be inserted.
        - mode (str): 'random' or 'selective'. In 'selective' mode, only selected_words are considered for insertion.
        - selected_words (list): List of words to consider for insertion in 'selective' mode.
        - print_info (bool): If True, prints information about the insertions.
        """
        # words = self.tokenizer(text)
        # n = max(1, int(p * len(words)))
        # new_words = words.copy()
        # for _ in range(n):
        #     random_word = random.choice(words)
        #     similars = self.get_similar_words(random_word)
        #     if similars:
        #         similar_word = random.choice(similars)
        #         insert_position = random.randint(0, len(new_words))
        #         new_words.insert(insert_position, similar_word)
        #         if print_info:
        #             print(f'Inserted {similar_word} at position {insert_position}')
        # return ' '.join(new_words)
        words = self.tokenizer(text)
        n = max(1, int(p * len(words)))
        assert mode in ['random', 'selective', 'given'], "mode must be 'random', 'selective' or 'given'"
        new_words = words.copy()
        insertion_res = []  
        if mode == 'random':
            for i in range(n):
                word_to_insert = self.add_word(new_words, mode)
                insertion_res.append(word_to_insert)
        else:
            if mode == 'selective':
  
                selected_words = list(set(selected_words).intersection(set(words)))
            random.shuffle(selected_words)
            if n > len(selected_words): 
                for given_word in selected_words:
                    word_to_insert = self.add_word(new_words, mode, given_word)
                    insertion_res.append(word_to_insert)
                for i in range(n - len(selected_words)):
                    word_to_insert = self.add_word(new_words, 'random')
                    insertion_res.append(word_to_insert)
            else: 
                for i in range(n):
                    word_to_insert = self.add_word(new_words, mode, selected_words[i])
                    insertion_res.append(word_to_insert)
        if print_info:
            print('insertion info:', insertion_res)
        return new_words

    def add_word(self, words, mode, given_word=None):
        random_idx = random.randint(0, len(words) - 1)
        if mode == 'given' and given_word is not None: 
            word_to_insert = given_word
            insert_pair = ('', given_word)
        elif mode == 'selective' and given_word is not None:  
            similars = self.get_similar_words(given_word)
            if len(similars) == 0:  
                return ('', '')
            word_to_insert = random.choice(similars)
            insert_pair = (given_word, word_to_insert)
        else:
            similars = []
            counter = 0
            while len(similars) < 1:
                random_word = words[random.randint(0, len(words) - 1)]
                similars = self.get_similar_words(random_word)
                counter += 1
                if counter >= 10:
                    return ('', '')
            word_to_insert = random.choice(similars)
            insert_pair = (random_word, word_to_insert)
        words.insert(random_idx, word_to_insert)
        return insert_pair

    def aug_by_swap(self, text, p, mode='random', selected_words=[], print_info=False):
        """
        Augments text by randomly swapping the positions of words.

        Parameters:
        - text (str): The input text to augment.
        - p (float): Proportion of words to be swapped.
        - print_info (bool): If True, prints information about the swaps.
        """
        # words = self.tokenizer(text)
        # n = max(1, int(p * len(words)))
        # new_words = words.copy()
        # for _ in range(n):
        #     idx1, idx2 = random.sample(range(len(new_words)), 2)

        #     new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        #     if print_info:
        #         print(f'Swapped positions: {idx1} and {idx2}')
        # return ' '.join(new_words)
        words = self.tokenizer(text)
        n = max(1, int(p * len(words)))
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        new_words = words.copy()
        swap_res = []
        if mode == 'random':
            for _ in range(n):
                new_words, swap_word = self.swap_word(new_words, mode)
                swap_res.append(swap_word)
        else:
            selected_words = list(set(selected_words).intersection(set(words)))

            random.shuffle(selected_words)
            if n > len(selected_words): 
                for selected_word in selected_words:
                    new_words, swap_word = self.swap_word(new_words, mode, selected_word)
                    swap_res.append(swap_word)
                for i in range(n - len(selected_words)):
                    new_words, swap_word = self.swap_word(new_words, 'random')
                    swap_res.append(swap_word)
            else:
                for i in range(n):
                    new_words, swap_word = self.swap_word(new_words, mode, selected_words[i])
                    swap_res.append(swap_word)
        if print_info:
            print('swap info:', swap_res)
        return new_words

    def swap_word(self, words, mode, selected_word=None):
        if mode == 'selective' and selected_word is not None and selected_word in words:
            idx_1 = words.index(selected_word) 
        else:
            idx_1 = random.randint(0, len(words) - 1)
        idx_2 = idx_1
        counter = 0
        while idx_2 == idx_1:
            idx_2 = random.randint(0, len(words) - 1)
            counter += 1
            if counter > 3:
                return words, ''
        words[idx_1], words[idx_2] = words[idx_2], words[idx_1]
        return words, words[idx_2] 

    def aug_by_deletion(self, text, p, mode='random',selected_words=[], print_info=False):
        """
        Augments text by randomly deleting words.

        Parameters:
        - text (str): The input text to augment.
        - p (float): Probability of each word being deleted.
        - print_info (bool): If True, prints information about the deletions.
        """
        # words = self.tokenizer(text)
        # new_words = [word for word in words if random.random() > p]
        # if print_info:
        #     deleted_words = set(words) - set(new_words)
        #     print('Deleted words:', deleted_words)
        # return ' '.join(new_words)
        words = self.tokenizer(text)
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        words_been_deleted = []
        if len(words) == 1:
            return words
        if mode == 'random':
            new_words = []
            for word in words:
                r = random.uniform(0, 1)
                if r > p:
                    new_words.append(word)
                else:
                    words_been_deleted.append(word)
        else: 
            selected_words = list(set(selected_words).intersection(set(words)))

            random.shuffle(selected_words)
            n = int(p * len(words))
            new_words = []
            for word in words:
                if word in selected_words and len(words_been_deleted) < n and word not in words_been_deleted:  # 最多删n个词, 控制每个词最多被删一次
                    words_been_deleted.append(word)
                    continue
                else:
                    new_words.append(word)
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]
        if print_info:
            print('deletion info:', words_been_deleted)
        return new_words

    def aug_by_selection(self, text, selected_words=[], print_info=False):
        """
        Augments text by selecting specific words from the text.

        Parameters:
        - text (str): The input text to augment.
        - selected_words (list): List of words to be selected.
        - print_info (bool): If True, prints information about the selections.
        """
        words = self.tokenizer(text)
        if len(selected_words) == 0:
            print('No selected words provided for:', words)
            return words
        new_words = []
        for w in words:
            if w in selected_words:
                if print_info:
                    print('selection info:', w)
                new_words.append(w)
        return new_words

    def random_text_augmentation(self, text, prob_dict=None, num_aug_dict=None,
                                 include_orig_sent=True, print_info=False):
        """
        Performs random augmentation using various techniques based on specified probabilities.

        Parameters:
        - text (str): The input text to augment.
        - prob_dict (dict): Dictionary specifying the probability for each augmentation type.
        - num_aug_dict (dict): Dictionary specifying the number of augmentations for each type.
        - include_orig_sent (bool): If True, includes the original sentence in the output.
        - print_info (bool): If True, prints detailed information about each augmentation.
        """
        if prob_dict is None:
            prob_dict = {'r': 0.1, 'i': 0.1, 's': 0.1, 'd': 0.1}
        if num_aug_dict is None:
            num_aug_dict = {'r': 1, 'i': 1, 's': 1, 'd': 1}

        augmented_texts = []
        method_list = []

        for _ in range(num_aug_dict['r']):
            augmented_texts.append(self.aug_by_replacement(text, prob_dict['r'], print_info=print_info))
            method_list.append('replacement')

        for _ in range(num_aug_dict['i']):
            augmented_texts.append(self.aug_by_insertion(text, prob_dict['i'], print_info=print_info))
            method_list.append('insertion')

        for _ in range(num_aug_dict['s']):
            augmented_texts.append(self.aug_by_swap(text, prob_dict['s'], print_info=print_info))
            method_list.append('swap')

        for _ in range(num_aug_dict['d']):
            augmented_texts.append(self.aug_by_deletion(text, prob_dict['d'], print_info=print_info))
            method_list.append('deletion')

        if include_orig_sent:
            augmented_texts.append(text)
            method_list.append('original')

        combined = list(zip(augmented_texts, method_list))
        shuffle(combined)
        augmented_texts, method_list = zip(*combined)

        return list(augmented_texts), list(method_list)

    def selective_text_augmentation(self, text, role_kws_dict, prob_dict=None, num_aug_dict=None,
                                    include_orig_sent=True, print_info=False):
        """
        Performs selective augmentation based on specified role keyword dictionaries.

        Parameters:
        - text (str): The input text to augment.
        - role_kws_dict (dict): Dictionary of role keywords used for different types of augmentations.
        - prob_dict (dict): Dictionary specifying the probability for each augmentation type.
        - num_aug_dict (dict): Dictionary specifying the number of augmentations for each type.
        - include_orig_sent (bool): If True, includes the original sentence in the output.
        - print_info (bool): If True, prints detailed information about each augmentation.
        """
        if prob_dict is None:
            prob_dict = {'r': 0.1, 'ii': 0.1, 'oi': 0.1, 's': 0.1, 'd': 0.1, 'sl': 0.1}
        if num_aug_dict is None:
            num_aug_dict = {'r': 1, 'ii': 1, 'oi': 1, 's': 1, 'd': 1, 'sl': 1}

        augmented_texts = []
        method_list = []

        # Replacement with class-indicating words
        for _ in range(num_aug_dict['r']):
            augmented_texts.append(self.aug_by_replacement(text, prob_dict['r'], mode='selective', selected_words=role_kws_dict['CW'], print_info=print_info))
            method_list.append('replacement')

        # Inner insertion (insert similar words of class words)
        for _ in range(num_aug_dict['ii']):
            augmented_texts.append(self.aug_by_insertion(text, prob_dict['ii'], mode='selective', selected_words=role_kws_dict['CW'], print_info=print_info))
            method_list.append('inner insertion')

        # Outer insertion (insert words from outside, e.g., noise words)
        for _ in range(num_aug_dict['oi']):
            augmented_texts.append(self.aug_by_insertion(text, prob_dict['oi'], mode='given', selected_words=role_kws_dict['FW_out'], print_info=print_info))
            method_list.append('outer insertion')

        # Swap with class-indicating words
        for _ in range(num_aug_dict['s']):
            augmented_texts.append(self.aug_by_swap(text, prob_dict['s'], mode='selective', selected_words=role_kws_dict['CW'], print_info=print_info))
            method_list.append('swap')

        # Deletion (delete fake class-indicating words or noise words)
        for _ in range(num_aug_dict['d']):
            augmented_texts.append(self.aug_by_deletion(text, prob_dict['d'], mode='selective', selected_words=role_kws_dict['FW_in'], print_info=print_info))
            method_list.append('deletion')

        # Selection (select class words to form a positive sample)
        for _ in range(num_aug_dict['sl']):
            augmented_texts.append(self.aug_by_selection(text, role_kws_dict['CW'], print_info=print_info))
            method_list.append('selection')

        if include_orig_sent:
            augmented_texts.append(text)
            method_list.append('original')

        combined = list(zip(augmented_texts, method_list))
        shuffle(combined)
        augmented_texts, method_list = zip(*combined)

        return list(augmented_texts), list(method_list)

# # Example usage
# if __name__ == "__main__":
#     ta = TextAugmenter('en')
#     sentence = 'The quick brown fox jumps over the lazy dog'
#     prob_dict = {'r': 0.2, 'i': 0.2, 's': 0.2, 'd': 0.2}
#     num_aug_dict = {'r': 2, 'i': 2, 's': 2, 'd': 2}
#     augmented_texts, methods = ta.random_text_augmentation(sentence, prob_dict, num_aug_dict, print_info=True)
#     for text, method in zip(augmented_texts, methods):
#         print(f"{method}: {text}")

