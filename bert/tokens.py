from typing import Optional, Union

import tokenizers
from tokenizers.models import WordLevel, TokenizedSequence, TokenizedSequenceWithOffsets
from tokenizers import Tokenizer, Encoding, AddedToken
from tokenizers.normalizers import Lowercase, Sequence, unicode_normalizer_from_str

from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import CharDelimiterSplit, WhitespaceSplit
from tokenizers.implementations import BaseTokenizer

from transformers import PreTrainedTokenizerFast
from typing import List, Optional, Union


class WordLevelTokenizer(BaseTokenizer):
    """ WordLevelBertTokenizer
    Represents a simple word level tokenization for BERT.
    """
    def __init__(self,
                 vocab_file: Optional[str] = None,
                 unk_token: Union[str, AddedToken] = "[UNK]",
                 sep_token: Union[str, AddedToken] = "[SEP]",
                 cls_token: Union[str, AddedToken] = "[CLS]",
                 pad_token: Union[str, AddedToken] = "[PAD]",
                 mask_token: Union[str, AddedToken] = "[MASK]",
                 lowercase: bool = False,
                 unicode_normalizer: Optional[str] = None, ):

        if vocab_file is not None:
            tokenizer = Tokenizer(WordLevel(vocab_file,
                                            unk_token=unk_token,
                                            # sep_token=sep_token,
                                            # cls_token=cls_token,
                                            # pad_token=pad_token,
                                            # mask_token=mask_token,
                                            ))
        else:
            tokenizer = Tokenizer(WordLevel())

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = WhitespaceSplit()
        # tokenizer.pre_tokenizer = CharDelimiterSplit(',')

        if vocab_file is not None:
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")

            tokenizer.post_processor = tokenizers.processors.BertProcessing(
                (str(sep_token), sep_token_id),
                (str(cls_token), cls_token_id)
                )

        parameters = {"model": "WordLevel",
                       "unk_token": unk_token,
                       "sep_token": sep_token,
                       "cls_token": cls_token,
                       "pad_token": pad_token,
                       "mask_token": mask_token,
                       "lowercase": lowercase,
                       "unicode_normalizer": unicode_normalizer,
                       }

        super().__init__(tokenizer, parameters)


class WordLevelBertTokenizer(PreTrainedTokenizerFast):

    def __init__(self, vocab, bos_token="[CLS]", eos_token="[SEP]", sep_token="[SEP]", cls_token="[CLS]",
            unk_token="[UNK]", pad_token="[PAD]", mask_token="[MASK]", **kwargs):
        # print('Start: create WordLevelTokenizer...')
        tokenizer = WordLevelTokenizer(vocab)
        # print('Finish: create WordLevelTokenizer...')
        super().__init__(tokenizer,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         cls_token=cls_token,
                         pad_token=pad_token,
                         mask_token=mask_token,
                         **kwargs, )

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """
        Copied from BertTokenizer
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[
        int]:
        """
        Copied from BertTokenizer
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
