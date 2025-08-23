import torch
import torch.nn as nn

"""
- Encoder: 과거 데이터 입력 → `nn.Linear(X_enc_features, d_model)`
- Decoder: 미래에 알려진 feature + 이전 시점 y_prev 입력 → `nn.Linear(X_dec_features + 1, d_model)`
- batch_first=True → 입력 shape [B, seq_len, feature]로 간단하게 처리.
- d_model=64, nhead=4, num_layers=2 등은 안정적인 학습을 위한 기본 설정.
- output_layer → Transformer 출력 → 최종 예측(y_full)으로 변환.
"""
class SimpleTransformer(nn.Module):
    def __init__(self, X_enc_features, X_dec_features, 
             num_stores, num_menus, num_store_menus, emb_dim=16,
             d_model=64, nhead=4, num_layers=2, pred_len=7, dropout=0.1):
        super().__init__()
        self.encoder_input = nn.Linear(X_enc_features + 3*emb_dim, d_model)
        self.decoder_input = nn.Linear(X_dec_features + 1 + 3*emb_dim, d_model) # known future + previous y_full + 임베딩

        self.store_emb = nn.Embedding(num_stores, emb_dim)
        self.menu_emb = nn.Embedding(num_menus, emb_dim)
        self.store_menu_emb = nn.Embedding(num_store_menus, emb_dim)

        # Dropout 적용 -> regularization
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, X_enc, X_dec_future, store_id, menu_id, store_menu_id, y_prev=None):
        B, dec_len, _ = X_dec_future.shape
        if y_prev is None:
            y_prev = torch.zeros(B, dec_len, device=X_enc.device)
        y_prev = y_prev.unsqueeze(-1)

        # Embedding lookup
        store_vec = self.store_emb(store_id).unsqueeze(1).repeat(1, X_enc.shape[1], 1)        # [B, enc_len, emb_dim]
        menu_vec = self.menu_emb(menu_id).unsqueeze(1).repeat(1, X_enc.shape[1], 1)
        store_menu_vec = self.store_menu_emb(store_menu_id).unsqueeze(1).repeat(1, X_enc.shape[1], 1)

        # Encoder input: feature + embeddings concat
        enc_input = torch.cat([X_enc, store_vec, menu_vec, store_menu_vec], dim=-1)
        enc_emb = self.encoder_input(enc_input)
        enc_out = self.encoder(enc_emb)

        # Decoder input: known future + prev y + embeddings
        store_vec_dec = store_vec[:, :dec_len, :]
        menu_vec_dec = menu_vec[:, :dec_len, :]
        store_menu_vec_dec = store_menu_vec[:, :dec_len, :]

        dec_input = torch.cat([X_dec_future, y_prev, store_vec_dec, menu_vec_dec, store_menu_vec_dec], dim=-1)
        dec_emb = self.decoder_input(dec_input)
        out = self.decoder(dec_emb, enc_out)
        out = self.output_layer(out).squeeze(-1)
        return out
