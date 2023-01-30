import random
import torch
import torch.nn as nn
import os

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers=4, dropout_ratio=0):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)

    # 인코더는 소스 문장을 입력으로 받아 문맥 벡터(context vector)를 반환        
    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보

        embedded = self.dropout(self.embedding(src))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [단어 개수, 배치 크기, 히든 차원]: 현재 단어의 출력 정보
        # hidden: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # cell: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        return hidden, cell # context vector 반환

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers=4, dropout_ratio=0):
        super().__init__()

        self.embedding = nn.Embedding(embed_dim, output_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input, hidden, cell):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현 (토큰)
        # hidden: [레이어 개수, 배치 크기, 히든 차원]
        # cell = context: [레이어 개수, 배치 크기, 히든 차원]

        input = input.unsqueeze(0)
        # input: [단어 개수 == 1, 배치 크기]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [단어 개수 = 1, 배치 크기, 히든 차원]: 현재 단어의 출력 정보
        # hidden: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # cell: [레이어 개수, 배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보

        # 단어 개수는 어차피 1개이므로 차원 제거
        prediction = self.fc_out(output.squeeze(0))
        # prediction : [배치 크기, 출력 차원]
        
        # (현재 출력 단어, 현재까지의 모든 단어의 정보, 현재까지의 모든 단어의 정보)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.device = device

    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        
        # 먼저 인코더를 거쳐 문맥 벡터(context vector)를 추출
        hidden, cell = self.encoder(src)

        trg_len = trg.shape[0] # 단어 개수
        batch_size = trg.shape[1] # 배치 크기
        trg_vocab_size = self.decoder.output_dim # 출력 차원

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 초기화
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        # range(0,trg_len)이 아니라 range(1,trg_len)인 이유 : 0번째 trg는 항상 <sos>라서 그에 대한 output도 항상 0이기 때문
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            # 디코더(decoder)의 최종 결과를 담을 텐서 객체 초기화# 랜덤 숫자가 teacher_forcing_ratio보다 작으면 True니까 teacher_force=1
            teacher_force = random.random() < teacher_forcing_ratio
            
            input = trg[t] if teacher_force else top1 # 현재의 출력 결과를 다음 입력에서 넣기
        
        return outputs

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= "5"  # Set the GPU 5 to use

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    torch.manual_seed(777)
    if device =='cuda':
        torch.cuda.manual_seed_all(777)

    layers = 4
    hidden_dim = 128
    input_dim = 100
    output_dim = 100

    # Encoder embedding dim
    enc_emb_dim = 256
    # Decoder embedding dim
    dec_emb_dim = 256

    hid_dim = 512
    n_layers = 4

    enc_dropout = 0.5
    dec_dropout = 0.5
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)

    model = Seq2Seq(enc, dec, device)
    print(model)
    
