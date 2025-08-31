from tinygrad import Tensor
import os

class SPieceTokenizer:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return SPieceTokenizer(path, **kwargs)

    def __init__(self, tokenizer_path, add_bos=False, add_eos=True):
        self.add_bos = add_bos
        self.add_eos = add_eos
        import sentencepiece
        if hasattr(tokenizer_path, 'numpy'):
            tokenizer_path = tokenizer_path.numpy().tobytes()

        if isinstance(tokenizer_path, bytes):
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_path, add_bos=self.add_bos, add_eos=self.add_eos)
        else:
            if not os.path.isfile(tokenizer_path):
                raise ValueError("invalid tokenizer")
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path, add_bos=self.add_bos, add_eos=self.add_eos)

    def get_vocab(self):
        out = {}
        for i in range(self.tokenizer.get_piece_size()):
            out[self.tokenizer.id_to_piece(i)] = i
        return out

    def __call__(self, string):
        out = self.tokenizer.encode(string)
        return {"input_ids": out}

    def serialize_model(self):
        import numpy as np
        return Tensor(np.array(list(self.tokenizer.serialized_model_proto()), dtype=np.uint8))
