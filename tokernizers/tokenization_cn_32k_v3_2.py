from collections import namedtuple
import os
import sentencepiece as spm
from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {
    'vocab_file': 'gpt2_huamei_corpus_bpe_32k_v3.2.model',
    #'vocab_file':'gpt2_huamei_corpus_bpe_24k.model'
    'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {

}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {

}
    

class Tokenization(object):
    """
    Tokenization object to hold tokenization, (processed text),and original
    text. Can hold tokenization as Ids or tokens.

    It also holds command tokens (pad, unk, etc.) for the tokenization.
    This allows functions to pad/operate on tokenizations without having
    access to the full tokenizer, just the tokenization.

    Several standard array operations are implemented (insert, append, extend).
    """
    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.asIds = asIds
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.asIds:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def insert(self, idx, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text = other.token + self.text
                self.original_text = other.token + self.original_text
            elif idx == len(self.tokenization)-1:
                self.text += other.token
                self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, list) and isinstance(other[0], (CommandToken, TypeToken)):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self

"""define some default command tokens for the tokenizer to use"""
token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))

def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]

class CommandToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))

DEFAULT_COMMAND_TOKENS = [
                            ('pad', 0),
                            ('eos', 1),
                            ('bos', 2),
                            ('unk', 3),
                            ('sep', 4),
                            ('L2R', 5),
                            ('ENC', 6),
                            ('MASK', 7),
]

DEFAULT_COMMAND_TOKENS = prep_command_tokens(DEFAULT_COMMAND_TOKENS)

SPECIAL_TOKEN_NUM = 5

BPE_ID2_CMD=[0,3,2,1,4]
CMD_ID2_BPE=[0,3,2,1,4]

"""define some default type tokens for bert training"""

TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))

def prep_type_tokens(tokenlist, token_format=token_format):
    return [TypeToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]

class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))

DEFAULT_TYPE_TOKENS = [
                            ('function', 0),
                            ('command', 1),
                            ('str0', 2),
                            ('str1', 3),
                            ('str2', 4),
                            ('embedding0', 5),
                            ('embedding1', 6),
                            ('embedding2', 7),
                            ('arg0', 8),
                            ('arg1', 9),
                            ('arg2', 10),
]
DEFAULT_TYPE_TOKENS = prep_type_tokens(DEFAULT_TYPE_TOKENS)

class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """
    def __init__(self, text_tokenizer, command_tokens=None, type_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)

        # set command tokens
        if command_tokens is None:
            command_tokens = DEFAULT_COMMAND_TOKENS
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self._command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

        # set type tokens
        if type_tokens is None:
            type_tokens = DEFAULT_TYPE_TOKENS
        self.type_tokens = type_tokens
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}
        if not hasattr(self, 'num_type_tokens'):
            self.num_type_tokens = len(self.type_tokens)

        # parse tokens and vocabs from tokenizer
        self._tokens = list(self.command_token_map.keys()) + list(self.text_tokenizer.tokens)
        self._vocab = {t:Id for Id,t in self.command_id_map.items()}
        self._vocab.update({t:Id+self.num_command_tokens for t,Id in self.text_tokenizer.vocab.items()})

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t:Id+self.num_command_tokens for t,Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t:Id for Id,t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t:Id for Id, t in self.type_id_map.items()}


    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def get_type(self, name):
        """get type token corresponding to `name`"""
        return self.type_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def token_types(self):
        """list (or iterable) of all token types for tokenizer"""
        return self._token_types

    @property
    def token_type_vocab(self):
        """dictionary mapping token types to ids for tokenizer"""
        return self._token_type_vocab

    @property
    def command_tokens(self):
        """list (or iterable) of all command tokens for tokenizer"""
        return self._command_token_tokens

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        tokenization = self.text_tokenizer.EncodeAsIds(text, process_fn=process_fn)
        tokenization.tokenization = [t+self.num_command_tokens for t in tokenization.tokenization]
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.text_tokenizer.EncodeAsTokens(text, process_fn=process_fn)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def IdToToken(self, Id, type_token=False):
        """convert Id to token accounting for command and type tokens"""
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id < self.num_command_tokens:
            return self.command_id_map[Id].token

        Id = Id - self.num_command_tokens

        if(Id<SPECIAL_TOKEN_NUM):
            print(f"CMD ID {Id} --> BPE {CMD_ID2_BPE[Id]}\n")
            Id = CMD_ID2_BPE[Id]

        return self.text_tokenizer.IdToToken(Id)

    def TokenToId(self, token, type_token=False):
        """convert token to Id accounting for command and type tokens"""
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        if token in self.command_token_map:
            return self.command_token_map[token].Id

        Id = self.text_tokenizer.TokenToId(token)

        if(Id<SPECIAL_TOKEN_NUM):
            #print(f"BPE ID {Id} --> CMD {BPE_ID2_CMD[Id]}\n")
            Id = BPE_ID2_CMD[Id]
            return Id

        else:

            return Id + self.num_command_tokens

    def DecodeIds(self, Ids, type_token=False):
        """
        convert Ids to tokens accounting for command and type tokens, tokens
        are joined and returned as a string.
        """
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        rtn_strs = []
        current_str = []
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif Id < self.num_command_tokens:
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id - self.num_command_tokens)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
        return ' '.join(rtn_strs)

    def DecodeTokens(self, Tokens, type_token=False):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        rtn_strs = []
        current_str = []
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        for t in Tokens:
            if isinstance(t, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif t in self.command_token_map:
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t)
            else:
                current_str.append(t)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
        return ' '.join(rtn_strs)

class GPT2Tokenizer_cn(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    #pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, errors='replace', unk_token="<unk>",
                 bos_token="<bos>", eos_token="<eos>", **kwargs):
        super(GPT2Tokenizer_cn, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

        text_tokenizer = SentencePieceTokenizer(vocab_size=32128, model_path=vocab_file,pad_token=0)
        self.tokenizer = Tokenizer(text_tokenizer)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def _tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space toto get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        if add_prefix_space:
            text = ' ' + text

        return self.tokenizer.text_tokenizer._tokenize(text)
        

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.tokenizer.TokenToId(token)
        

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.tokenizer.IdToToken(index)
        

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        return text

    def save_vocabulary(self, save_directory):        
        print("save_vocabulary NotImplementedError")

        return os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"]), os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])
        
class TextTokenizer(object):
    """
    Interface for text tokenizer
    """
    def __init__(self):
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = 0
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_text_tokens

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn)

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        raise NotImplementedError('TextTokenizer tokens property not implemented')

    @property
    def vocab(self):
        """dictionary mapping tokens to ids"""
        raise NotImplementedError('TextTokenizer vocab property not implemented')

    @staticmethod
    def exists(model_path):
        """check if the filepath for a text tokenizer exists"""
        raise NotImplementedError('TextTokenizer exists method not implemented')

    def Train(self, corpus):
        """train a tokenizer on a data corpus and save model for future use"""
        raise NotImplementedError('TextTokenizer Train not implemented')

    def EncodeAsIds(self, text, process_fn=None):
        """
        Preprocess text and encode as ids. Return a tokenization object with
        original text, processed text, and id tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsIds not implemented')

    def EncodeAsTokens(self, text, process_fn=None):
        """
        Preprocess text and encode as tokens. Return a tokenization object with
        original text, processed text, and token tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsTokens not implemented')

    def IdToToken(self, Id):
        """Convert an Id to Token. Reverse lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer IdToToken not implemented')

    def TokenToId(self, token):
        """Convert a Token to Id. Lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer TokenToId not implemented')

    def DecodeIds(self, Ids):
        """Convert a list or tokenization object of Ids to a text string"""
        raise NotImplementedError('TextTokenizer DecodeIds not implemented')

    def DecodeTokens(self, Tokens):
        """Convert a list or tokenization object of tokens to a text string"""
        raise NotImplementedError('TextTokenizer DecodeTokens not implemented')


class SentencePieceTokenizer(TextTokenizer):
    """Trains and uses sentencepiece for text tokenization"""
    def __init__(self, vocab_size=None, model_path=None, **kwargs):

        self.spm_model = model_path
        self.num_text_tokens = vocab_size

        self._tokens = []
        self._vocab = {}
        self.load_spm_model()
        super(SentencePieceTokenizer, self).__init__()

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    @staticmethod
    def exists(model_path):
        if model_path is None:
            return False
        # check if path exists
        dne = not os.path.exists(model_path)
        # check if path.model exists
        if dne and not model_path.endswith('.model'):
            dne = not os.path.exists(model_path+'.model')
        return not dne

    def load_spm_model(self):
        """load sentencepiece model and parse vocab"""
        if not os.path.exists(self.spm_model) and not self.spm_model.endswith('.model'):
            self.spm_model = self.spm_model+'.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.spm_model)
        self.vocab_size = self.num_text_tokens = len(self.sp)
        self._tokens = [self.IdToToken(t) for t in range(self.vocab_size)]
        self._vocab = {t: i for i,t in enumerate(self._tokens)}

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to sentencepiece Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsIds(processed_text)
        return Tokenization(tokens, processed_text, text)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert text to sentencepiece tokens"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsTokens(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id):
        """convert Id to sentencpiece token"""
        return self.sp.IdToPiece(Id)

    def TokenToId(self, token):
        """convert sentencpiece token to Id"""
        Id = self.sp.PieceToId(token)
        return Id

    def DecodeIds(self, Ids):
        """converts ids to a text string"""
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return self.sp.DecodeIds(Ids)

    def DecodeTokens(self, Tokens):
        """converts sentencepiece tokens to a text string"""
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.sp.DecodeTokens(Tokens)

    def _tokenize(self, text):
        return self.sp.encode_as_pieces(text)


class GPT2BPETokenizer_CN(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, errors='replace', unk_token="<unk>",
                 bos_token="<bos>", eos_token="<eos>", **kwargs):
        super(GPT2BPETokenizer_CN, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def _tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space toto get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        if add_prefix_space:
            text = ' ' + text

        return self.tokenizer.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.tokenizer.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.tokenizer.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        return text

    def save_vocabulary(self, save_directory):
        print("save_vocabulary NotImplementedError")

        return os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"]), os.path.join(save_directory,
                                                                                           VOCAB_FILES_NAMES[
                                                                                               "merges_file"])
