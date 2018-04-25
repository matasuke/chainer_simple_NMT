import argparse
import pickle
from tqdm import tqdm

class Tokenizer:
    def __init__(self, lang='jp', to_lower=True, remove_suffix=True):
        self.lang = lang
        self.to_lower = to_lower
        self.remove_suffix = remove_suffix

        if lang == 'jp':
            from janome.tokenizer import Tokenizer
            self.t = Tokenizer()
            self.segmenter = lambda sentence: list(token.surface for token in self.t.tokenize(sentence))

        elif lang == 'ch':
            import jieba
            self.segmenter = lambda sentence: list(jieba.cut(sentence))

        elif lang == 'en':
            import nltk
            self.segmenter = lambda sentence: list(nltk.word_tokenize(sentence))

    def pre_process(self, sentence):
        if self.to_lower:
            sentence = sentence.strip().lower()
        if self.remove_suffix and sentence[-1] in ('.', '。'):
            sentence = sentence[0: -1]
        return self.segmenter(sentence)

def token2index(tokens, word_ids):
    return [ word_ids[token] if token in word_ids else word_ids['<UNK>'] for token in tokens ]

def load_pickle(in_file):
    with open(in_file, 'rb') as f:
        row_data = pickle.load(f)
    return row_data

def save_pickle(in_file, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(in_file, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in_path', type=str)
    parser.add_argument('-out_path', type=str)
    parser.add_argument('--lang', type=str, choices=['jp', 'en', 'ch'])
    parser.add_argument('--tolower', action='store_true')
    parser.add_argument('--remove_suffix', action='store_true')
    parser.add_argument('--cutoff', type=int, default=5)
    parser.add_argument('--tonumpy', action='store_true')
    args = parser.parse_args()

    tokenizer = Tokenizer(lang=args.lang, to_lower=args.tolower, remove_suffix=args.remove_suffix)

    sentence_idx = 0
    sentences = []

    word_counter = {}
    word_ids = {
        '<SOS>': 0,
        '<EOS>': 1,
        '<UNK>': 2,
    }
    
    f = open(args.in_path, 'r')
    lines = f.readlines()
    #tokenize sentences
    for line in tqdm(lines):
        sentence_tokens = ['<SOS>']
        sentence_tokens += tokenizer.pre_process(line)
        sentence_tokens += ['<EOS>']
        
        sentences.append({'sentence': line.strip(), 'tokens': sentence_tokens, 'encoded_sentence': sentence_tokens, 'sentence_idx': sentence_idx})
        sentence_idx += 1


    #create vocabrary dict
    for sentence in tqdm(sentences):
        tokens = sentence['tokens']

        for token in tokens:
            if token in word_counter:
                word_counter[token] += 1
            else:
                word_counter[token] = 1
    
    print("total distinct words:{0}".format(len(word_counter)))
    print('top 30 frequent words:')
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    for word, num in sorted_words[:30]:
        print('{0} - {1}'.format(word, num))

    for word, num in tqdm(word_counter.items()):
        if num > args.cutoff and word not in word_ids:
            word_ids[word] = len(word_ids)

    print('total distinct words except words less than {0}: {1}'.format(args.cutoff, len(word_ids)))

    #encoding
    for sentence in tqdm(sentences):
        sentence['encoded_sentence'] = token2index(sentence['encoded_sentence'], word_ids)

    output_dataset = {}
    output_dataset['train'] = {'sentences': sentences, 'word_ids': word_ids}
    
    save_pickle(output_dataset, args.out_path)
