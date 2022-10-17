from BPE.bpe_encoder import TextEncoder


class TokenIndexer:
    def __init__(self):
        """
        Including word tokens, special tokens, dialogue states, and emotion indices
        """
        self.text_encoder = TextEncoder()

        # Special tokens (Start / End of continguous dialogue sentence, Padding)
        self.PAD_IDX = 0
        self.SOS_IDX = len(self.text_encoder.encoder) + 1
        self.EOS_IDX = len(self.text_encoder.encoder) + 2
        
        # Size of vocabulary and special tokens 
        self.vocab_size = len(self.text_encoder.encoder) + 3
        
        # Dialog state indices
        self.DS_SPEAKER_IDX = 1
        self.DS_LISTENER_IDX = 2
        
        # Emotion label map
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 
            'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9,
            'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 
            'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31
        }

    def encode_text(self, x):
        # from text to indexs
        return self.text_encoder.encode(x)

    def decode_index2text(self, idx):
        # from indexs to text (indexs range in [0,n_vocab+2) (SOS, EOS))
        if idx == self.SOS_IDX:
            return '[SOS]'
        elif idx == self.EOS_IDX:
            return '[EOS]'
        else:
            return self.text_encoder.decoder[idx]
