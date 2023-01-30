# Sequence to Sequence Learning with Neural Networks
https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf

## 구조

![enc-dec](https://user-images.githubusercontent.com/73745836/215377787-4295eb14-1ce7-402b-9baa-f783b86a9247.jpg)

![seq2seq](https://user-images.githubusercontent.com/73745836/215377760-8e4a4d3f-8311-47b6-b426-41be2e328ef0.jpg)

## 구현
- [torch](./Seq2Seq/Seq2Seq.py)

## 결과

```
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(100, 256)
    (lstm): LSTM(256, 512, num_layers=4, dropout=0.5)
  )
  (decoder): Decoder(
    (embedding): Embedding(256, 100)
    (lstm): LSTM(256, 512, num_layers=4, dropout=0.5)
    (fc): Linear(in_features=512, out_features=100, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```
