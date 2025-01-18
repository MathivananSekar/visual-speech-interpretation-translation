class Vocab:
    """
    Basic vocabulary class for mapping tokens to IDs and vice versa.
    """
    def __init__(self, tokens=None, specials=None):
        """
        tokens: a list of unique tokens in your vocabulary (e.g., words or characters).
        specials: a dict of special tokens: { 'pad': '<pad>', 'unk': '<unk>', ... }
        
        Example usage:
            Vocab(tokens=['hello','world'],
                  specials={'pad':'<pad>','unk':'<unk>','sos':'<sos>','eos':'<eos>'})
        """
        if tokens is None:
            tokens = []
        if specials is None:
            specials = {}

        # Keep track of special tokens in a specific order so they get consistent IDs.
        self.special_tokens = []
        for key in ['pad', 'unk', 'sos', 'eos']:
            if key in specials:
                self.special_tokens.append(specials[key])

        # Build the final list of all tokens: specials first, then normal tokens
        unique_tokens = list(dict.fromkeys(self.special_tokens + tokens))  # preserve order
        self.stoi = {tok: i for i, tok in enumerate(unique_tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

        # Store special token IDs
        self.pad_id = self.stoi[specials['pad']] if 'pad' in specials else None
        self.unk_id = self.stoi[specials['unk']] if 'unk' in specials else None
        self.sos_id = self.stoi[specials['sos']] if 'sos' in specials else None
        self.eos_id = self.stoi[specials['eos']] if 'eos' in specials else None

    def __len__(self):
        return len(self.stoi)

    def token_to_id(self, token):
        """
        Return the ID of a token. If not found, return <unk> ID.
        """
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.unk_id

    def id_to_token(self, idx):
        """
        Return the token corresponding to an ID.
        """
        return self.itos.get(idx, '<unk>')
