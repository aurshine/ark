from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_x, dec_state)