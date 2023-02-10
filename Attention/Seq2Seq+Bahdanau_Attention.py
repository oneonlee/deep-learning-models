import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, enc_embed_dim, enc_hid_dim, dropout_ratio=0, n_layers=1, isBidirectional=True):
        assert isBidirectional==True, "isBidirectional가 True일 때만 동작합니다! (Encoder에 사용되는 RNN은 양방향 RNN)"
        assert n_layers==1, "Encoder : n_layers가 1일 때만 동작합니다! (깊은 RNN을 사용하는 기능은 추후 추가 예정)"
        
        super().__init__()

        self.embedding = nn.Embedding(input_dim, enc_embed_dim)
        self.bi_rnn = nn.RNN(input_size=enc_embed_dim, hidden_size=enc_hid_dim, 
                             dropout=dropout_ratio, num_layers=n_layers,
                             bidirectional=isBidirectional)
        # self.num_directions = 2 if isBidirectional else 1
        self.dropout = nn.Dropout(dropout_ratio)
        

    def forward(self, src):
        # src: [seq_len, batch_size]: 각 단어의 인덱스(index) 정보

        embedded = self.dropout(self.embedding(src))
        # embedded: [seq_len, batch_size, enc_embed_dim]
        
        outputs, hidden = self.bi_rnn(embedded)
        # outputs: [seq_len, batch_size, num_directions * enc_hid_dim]: 현재 단어의 출력 정보
        #    -> 각 timestep마다 top layer의 hidden state로써, Attention Score 계산 시 사용됨
        # hidden: [num_directions * num_layers, batch_size, enc_hid_dim]: 현재까지의 모든 단어의 정보
        #    -> Decoder의 초기 hidden state로 사용됨
        
        return outputs, hidden
      
class Decoder(nn.Module):
    def __init__(self, output_dim, dec_embed_dim, dec_hid_dim, dropout_ratio=0, n_layers=1, isBidirectional=False):
        assert isBidirectional==False, "isBidirectional가 False일 때만 동작합니다! (Decoder에 사용되는 RNN은 단순 RNN)"
        assert n_layers==1, "Decoder : n_layers가 1일 때만 동작합니다! (깊은 RNN을 사용하는 기능은 추후 추가 예정)"
        
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, dec_embed_dim)
        self.rnn = nn.RNN(input_size=dec_embed_dim, hidden_size=dec_hid_dim, 
                          dropout=dropout_ratio, num_layers=n_layers, 
                          bidirectional=isBidirectional)
        self.fc = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)

        
    def forward(self, input, hidden):
        # input: [seq_len==1, batch_size]: seq_len == 단어 개수 == 1, 단어의 개수는 항상 1개이도록 구현 (토큰)
        # hidden: [1 * num_layers, batch_size, dec_hid_dim]

        embedded = self.dropout(self.embedding(input))
        # embedded: [seq_len==1, batch_size, dec_embed_dim]
        
        output, hidden = self.rnn(embedded, hidden)
        # output: [seq_len==1, batch_size, dec_hid_dim]: 현재 단어의 출력 정보
        # hidden: [1 * num_layers, batch_size, dec_hid_dim]: 현재까지의 모든 단어의 정보

        # 단어 개수는 어차피 1개이므로 차원 제거
        prediction = self.fc(output.squeeze(0))
        # prediction : [batch_size, dec_hid_dim]
        
        # (현재 출력 단어, 현재까지의 모든 단어의 정보)
        return prediction, hidden
      
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.Wb = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.Wc = nn.Linear(2 * enc_hid_dim, dec_hid_dim, bias=False)
        
        self.Wa_T = nn.Linear(dec_hid_dim, 1, bias=False)
        
        
    def forward(self, dec_prev_hidden, enc_outputs):
        # dec_prev_hidden : [1 * num_layers == 1, batch_size, dec_hid_dim]
        # enc_outputs : [src_seq_len, batch_size, num_directions * enc_hid_dim]
        
        src_seq_len = enc_outputs.shape[0]
        
        s_t_1 = dec_prev_hidden.repeat(src_seq_len, 1, 1) # [src_seq_len, batch_size, dec_hid_dim]
        weighted_s_t_1 = self.Wb(s_t_1) # [src_seq_len, batch_size, dec_hid_dim]
        weighted_H = self.Wc(enc_outputs) # [src_seq_len, batch_size, dec_hid_dim]
        
        attention_score = self.Wa_T(torch.tanh(weighted_s_t_1 + weighted_H))
        print("Attention Score", attention_score.shape) # [src_seq_len, batch_size, 1]
        
        attention_distribution = F.softmax(attention_score)
        print("Attention Distribution", attention_distribution.shape) # [src_seq_len, batch_size, 1]
        
        # Batch First == True
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs : [src_seq_len, batch_size, num_directions * enc_hid_dim] 
        #            -> [batch_size, src_seq_len, num_directions * enc_hid_dim]
        attention_distribution = attention_distribution.permute(1, 2, 0)
        # attention_distribution : [src_seq_len, batch_size, 1]
        #                       -> [batch_size, 1, src_seq_len]
        
        # torch.bmm: Batch Matrix Multiplication
        #     ex> [B, n, m] * [B, m, p] = [B, n, p]
        context_vector = torch.bmm(attention_distribution, enc_outputs)
        print("Context Vector", context_vector.shape) # (batch_size, 1, num_directions * enc_hid_dim)
        
        # Batch First == False
        context_vector = context_vector.permute(1, 0, 2)
        print("Context Vector", context_vector.shape) # (1, batch_size, num_directions * enc_hid_dim)
        
        return context_vector
      
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention, enc_hid_dim, dec_hid_dim, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        
        # Encoder의 양방향 RNN hidden state들을 합친 후, Decoder의 hidden dimmension으로 변환하는 FC 레이어
        self.fc = nn.Linear(2 * enc_hid_dim, dec_hid_dim)
        
        self.device = device

        
    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [seq_len, batch_size]
        # trg: [seq_len, batch_size]
        trg_len = trg.shape[0] # seq_len of trg
        batch_size = trg.shape[1] # batch_size
        trg_vocab_size = self.decoder.output_dim # 출력 차원
        
        # 먼저 Encoder를 거쳐, outputs과 Encoder의 hidden state를 추출
        enc_outputs, enc_hidden = self.encoder(src)
        print(enc_outputs.shape) # [seq_len, batch_size, num_directions * enc_hid_dim]
        print(enc_hidden.shpae) # [num_directions * num_layers, batch_size, enc_hid_dim]

        concatenated_hidden = torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=-1) # [batch_size, 2 * enc_hid_dim]
        dec_init_hidden = torch.tanh(self.fc(concatenated_hidden))
        # Decoder의 초기 hidden state는 Encoder의 최종 hidden state가 사용됨
        # 이때 hidden state의 차원을 2*encoder_hidden_dim에서 decoder_hidden_dim으로 변환시킴 
        print(dec_init_hidden.shape) # [batch_size, dec_hid_dim]

        # Decoder의 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :].unsqueeze(0)
        # Decoder의 최종 결과를 담을 텐서 객체 초기화
        dec_outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 타겟 단어의 개수만큼 반복하여 Decoder에 포워딩(forwarding)
        # range(0, trg_len)이 아니라 range(1, trg_len)인 이유 
        #     -> 0번째 trg는 항상 <sos>라서 그에 대한 output도 항상 0이기 때문
        dec_hidden = dec_init_hidden
        for t in range(1, trg_len):
            output, dec_hidden = self.decoder(input, dec_hidden)

            dec_outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력 Y를 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1 # 현재의 출력 결과를 다음 입력에서 넣기
            
            context_vector = attention(hidden, enc_outputs)
            input = torch.cat((input, context_vector), dim=2) # [1, batch_size, emb_dim + 2 * enc_hid_dim]
            
        return dec_outputs
      
      
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= "4"  # Set the GPU 4 to use

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    torch.manual_seed(777)
    if device =='cuda':
        torch.cuda.manual_seed_all(777)

    INPUT_DIM = 24745
    OUTPUT_DIM = 8854

    ENC_EMB_DIM = 512 # Encoder embedding dim
    DEC_EMB_DIM = 256 # Decoder embedding dim
    ENC_HID_DIM = 128 # Encoder hidden dim
    DEC_HID_DIM = 64 # Decoder hidden dim
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0
    
    N_LAYERS = 1
    
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, dropout_ratio=ENC_DROPOUT, n_layers=N_LAYERS, isBidirectional=True)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, dropout_ratio=DEC_DROPOUT, n_layers=N_LAYERS, isBidirectional=False)
        
    attention = BahdanauAttention(ENC_HID_DIM, DEC_HID_DIM)
    
    model = Seq2Seq(encoder, decoder, attention, ENC_HID_DIM, DEC_HID_DIM, device).to(device)
    
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters.') # The model has 15,733,526 trainable parameters.
