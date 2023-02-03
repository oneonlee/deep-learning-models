# Recurrent Neural Networks

### 수식

메모리 셀에서 은닉 상태를 계산하는 식을 다음과 같이 정의하고 구현하였음

$a_{t} = x_{t}U + h_{t−1}W + b$<br>
$h_{t} = \tanh{(a_{t})} = \tanh{(x_{t}U + h_{t−1}W + b)}$<br>
$o_{t} = h_{t}V + c$

## PyTorch를 통한 구현
- [Python file](./RNN.py)
- [ipynb file](./RNN.ipynb)




### PyTorch에서 제공하는 `nn.RNNCell` 코드 예제
```python
>>> rnn = nn.RNNCell(input_size, hidden_size) # (I, H)
>>> input = torch.randn(seq_len, 3, input_size) # (L, B, I)
>>> hx = torch.randn(3, hidden_size) # (B, H)
>>> output = []
>>> for i in range(seq_len): 
...     hx = rnn(input[i], hx) 
...     output.append(hx) # (L, B, H)
```

### PyTorch에서 제공하는 `nn.RNN` 코드 예제
```python
>>> rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False) # (I, H, num_layers)
>>> input = torch.randn(seq_len, 3, input_size) # (L, B, I)
>>> h0 = torch.randn(num_layers, 3, hidden_size) # (num_layers, B, H)
>>> output, hn = rnn(input, h0)
>>> output.shape # (L, B, H)
```
