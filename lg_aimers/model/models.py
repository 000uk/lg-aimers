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
                 num_store_menus, num_menus, emb_dim=16,
                 d_model=64, nhead=4, num_layers=2, pred_len=7, dropout=0.5):
        super().__init__()
        # + 2*emb_dim : store_menu_emb + menu_emb
        self.encoder_input = nn.Linear(X_enc_features + 2*emb_dim, d_model)
        self.decoder_input = nn.Linear(X_dec_features + 1 + 2*emb_dim, d_model)  # known future + y_prev + 임베딩

        self.store_menu_emb = nn.Embedding(num_store_menus, emb_dim)
        self.menu_emb = nn.Embedding(num_menus, emb_dim)

        # Dropout 적용 -> regularization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, X_enc, X_dec_future, store_menu_id, menu_id, y_prev=None):
        B, dec_len, _ = X_dec_future.shape
        enc_len = X_enc.shape[1]

        if y_prev is None:
            y_prev = torch.zeros(B, dec_len, device=X_enc.device)
        y_prev = y_prev.unsqueeze(-1)  # [B, dec_len, 1]

        # --------- Embedding lookup ---------
        store_menu_vec = self.store_menu_emb(store_menu_id)  # [B, emb_dim]
        menu_vec = self.menu_emb(menu_id)                    # [B, emb_dim]

        # ----- Encoder -----
        store_menu_vec_enc = store_menu_vec.unsqueeze(1).repeat(1, enc_len, 1)  # [B, enc_len, emb_dim]
        menu_vec_enc = menu_vec.unsqueeze(1).repeat(1, enc_len, 1)              # [B, enc_len, emb_dim]
        enc_input = torch.cat([X_enc, store_menu_vec_enc, menu_vec_enc], dim=-1)
        enc_emb = self.encoder_input(enc_input)
        enc_out = self.encoder(enc_emb)

        # ----- Decoder -----
        store_menu_vec_dec = store_menu_vec.unsqueeze(1).repeat(1, dec_len, 1)  # [B, dec_len, emb_dim]
        menu_vec_dec = menu_vec.unsqueeze(1).repeat(1, dec_len, 1)              # [B, dec_len, emb_dim]
        dec_input = torch.cat([X_dec_future, y_prev, store_menu_vec_dec, menu_vec_dec], dim=-1)
        dec_emb = self.decoder_input(dec_input)
        out = self.decoder(dec_emb, enc_out)   # [B, dec_len, d_model]
        out = self.output_layer(out).squeeze(-1)  # [B, dec_len]
        return out