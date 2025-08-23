import torch
import torch.nn as nn

"""
시계열 데이터에 Transformer 모델을 적용하는 다양한 전략과 하이퍼파라미터 설정에 대한 메모 또는 아이디어를 정리한 것입니다. 이 아이디어들은 모델의 학습 효율과 안정성을 높이기 위한 여러 가지 접근법을 설명하고 있습니다.

핵심적인 내용을 한국어로 풀어서 설명해 드릴게요.

1. Linear 레이어 in_features 자동 계산
Encoder: 인코더의 입력 레이어(nn.Linear)는 과거 데이터(X_enc)의 특징(feature) 개수를 입력으로 받습니다. 즉, X_enc.shape[-1]이 바로 in_features가 됩니다.

Decoder: 디코더의 입력 레이어는 미래에 알려진 특징과 이전 시점의 예측값을 함께 받습니다. 따라서 입력 특징의 개수는 X_dec_future.shape[-1] (미래 특징)에 +1 (예측하려는 값, 즉 y_prev)을 더한 값이 됩니다. 이는 디코더가 자기회귀(auto-regressive) 방식으로 다음 값을 예측하는 구조를 의미합니다.

2. y_full 타깃 학습 및 Teacher Forcing
Residual 대신 y_full 타깃 학습: 일반적으로 시계열 데이터는 trend, seasonal, residual로 분해하여 residual만 예측하는 방식을 사용합니다. 하지만, 이 메모는 모델이 처음부터 분해된 residual이 아닌, residual + trend + seasonal을 모두 합친 원본 데이터(y_full)를 직접 예측하도록 시도하는 아이디어입니다.

장점: 초기 학습 시 모델이 예측해야 할 정보가 더 풍부해져 학습이 더 안정적으로 진행될 수 있습니다. 특히 trend와 seasonal 정보가 모델의 예측에 중요한 역할을 할 수 있습니다.

Decoder Teacher Forcing (티처 포싱): Transformer 디코더는 예측 시점의 이전 값(즉, y_prev)을 입력으로 받아서 다음 값을 예측합니다.

티처 포싱이란? 학습 시점에 모델이 예측한 이전 값이 아닌, 실제 정답(ground truth)인 이전 값을 디코더의 다음 입력으로 넣어주는 기법입니다.

효과: 이 방식을 사용하면 학습 초기 단계에서 모델의 불안정한 예측이 누적되어 성능이 나빠지는 것을 막고, 안정적으로 수렴하도록 도와줍니다.

3. Transformer 구조 및 하이퍼파라미터
Transformer 구조 단순화: 복잡한 구조보다는 기본적인 인코더-디코더 구조를 사용하되, in_features를 자동으로 계산하는 리니어 레이어를 사용하고, d_model을 64 정도로 설정하는 것을 제안합니다. 이는 모델의 크기를 적절하게 유지하면서 효율적인 학습을 목표로 합니다.

batch_first=True: PyTorch의 nn.Transformer 모듈에 batch_first=True 옵션을 설정하면, 입력 텐서의 형태를 [배치 크기, 시퀀스 길이, 특징 개수] ([B, seq_len, feature])로 맞춰줄 수 있어 코드를 더 직관적으로 작성할 수 있습니다.

하이퍼파라미터: batch_size를 32, learning_rate를 1e-3으로 설정하는 것은 딥러닝에서 일반적인 시작 값입니다. 이 값들로 초기 실험을 진행하고, 필요에 따라 조정할 수 있습니다.

요약
제공된 메모는 **"어떻게 하면 Transformer 모델로 시계열 예측을 더 잘 할 수 있을까?"**에 대한 고민을 담고 있습니다.

목표: residual만 예측하는 대신, y_full을 직접 예측해 보자.

전략:

디코더에 미래에 알려진 정보와 이전 시점의 정답 값을 함께 넣어주자 (Teacher Forcing).

이렇게 하면 모델이 예측해야 할 정보가 더 풍부해지고, 학습이 안정적으로 이루어질 것이다.

기술적 구현: nn.Linear의 입력 차원을 동적으로 계산하고, batch_first=True를 사용해 코드를 간결하게 만들자.
"""
class SimpleTransformer(nn.Module):
    def __init__(self, X_enc_features, X_dec_features, d_model=64, nhead=4, num_layers=2, pred_len=7, dropout=0.1):
        super().__init__()
        self.encoder_input = nn.Linear(X_enc_features, d_model)
        self.decoder_input = nn.Linear(X_dec_features + 1, d_model)  # known future + previous y_full

        # Dropout 적용 -> regularization
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