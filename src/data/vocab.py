class Vocab:
    """
    Basic vocabulary class for mapping tokens to IDs and vice versa.
    """
    def __init__(self, tokens=None, specials=None):
        if tokens is None:
            tokens = []
        if specials is None:
            specials = {'pad': '<pad>', 'unk': '<unk>', 'sos': '<sos>', 'eos': '<eos>', 'blank': '<blank>'}

        self.special_tokens = []
        for key in ['pad', 'unk', 'sos', 'eos']:
            if key in specials:
                self.special_tokens.append(specials[key])

        unique_tokens = list(dict.fromkeys(self.special_tokens + tokens))
        self.stoi = {tok: i for i, tok in enumerate(unique_tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

        self.pad_id = self.stoi[specials['pad']] if 'pad' in specials else None
        self.unk_id = self.stoi[specials['unk']] if 'unk' in specials else None
        self.sos_id = self.stoi[specials['sos']] if 'sos' in specials else None
        self.eos_id = self.stoi[specials['eos']] if 'eos' in specials else None

    def __len__(self):
        return len(self.stoi)

    def token_to_id(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.unk_id

    def id_to_token(self, idx):
        return self.itos.get(idx, '<unk>')
