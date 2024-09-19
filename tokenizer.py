

CTC_BLANK = '<BLANK>'
OOV_TOKEN = '<OOV>'

class Tokenizer:
    def __init__(self, alphabet):
        self.char_map = {CTC_BLANK: 0, OOV_TOKEN: 1}
        for i, char in enumerate(alphabet):
            self.char_map[char] = i + 2
        self.rev_char_map = {v: k for k, v in self.char_map.items()}

    def encode(self, word_list):
        enc_words = []
        for word in word_list:
            enc_words.append([self.char_map.get(char, self.char_map[OOV_TOKEN]) for char in word])
        return enc_words

    def decode(self, enc_word_list, merge_repeated=True):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            prev_char = None
            for char_enc in word:
                char = self.rev_char_map.get(char_enc, OOV_TOKEN)
                if char != CTC_BLANK and char != OOV_TOKEN:
                    if not (merge_repeated and char == prev_char):
                        word_chars += char
                prev_char = char
            dec_words.append(word_chars)
        return dec_words

