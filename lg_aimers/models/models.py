import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, X_enc_features, X_dec_features, d_model=64, nhead=4, num_layers=2, pred_len=7, dropout=0.1):
        super().__init__()
        self.encoder_input = nn.Linear(X_enc_features, d_model)
        self.decoder_input = nn.Linear(X_dec_features + 1, d_model)  # known future + previous y_full

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, X_enc, X_dec_future, y_prev=None):
        B, dec_len, _ = X_dec_future.shape
        if y_prev is None:
            y_prev = torch.zeros(B, dec_len, device=X_enc.device)
        y_prev = y_prev.unsqueeze(-1)

        dec_input = torch.cat([X_dec_future, y_prev], dim=-1)  # [B, pred_len, n_features+1]

        enc_emb = self.encoder_input(X_enc)
        enc_out = self.encoder(enc_emb)

        dec_emb = self.decoder_input(dec_input)
        out = self.decoder(dec_emb, enc_out)
        out = self.output_layer(out).squeeze(-1)  # [B, pred_len]
        return out