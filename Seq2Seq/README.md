# Sequence to Sequence Learning with Neural Networks
https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf

## 구조

![enc-dec](https://user-images.githubusercontent.com/73745836/215377787-4295eb14-1ce7-402b-9baa-f783b86a9247.jpg)
![seq2seq](https://user-images.githubusercontent.com/73745836/215519704-c872af10-35af-4a0a-bcb1-d1c8cd14567e.jpg)

## 구현
- [torch](./Seq2Seq.py)

## 결과

```
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(24745, 512)
    (lstm): LSTM(512, 256, num_layers=4, dropout=0.5)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Decoder(
    (embedding): Embedding(8854, 256)
    (lstm): LSTM(256, 256, num_layers=4, dropout=0.5)
    (fc): Linear(in_features=256, out_features=8854, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```

```
The model has 21,684,374 trainable parameters.
```
