# autoformer_pytorch

This is an unofficial reproduction of [autoformer](https://arxiv.org/abs/2106.13008).
Official code is [here](https://github.com/thuml/Autoformer).

Depenedence：pytorch,einops

# run


```bash
git clone https://github.com/celtics1863/autoformer_pytorch
cd autoformer_pytorch
python ETTmtrain.py
```

# Results:

config settings:

- datasets: ETTm
- pred_len: 96
- label_len:96
- input_len:96

| model  | mae| rmse|
| --- |--- |---- |
| informer (official)  | 0.360 | 0.239  |
| autoformer (official)| 0.301 | 0.218  |
| autoformer (ours) | 0.291 | 0.175 |


# Defference and Idea

1. We set if no time stamp inputs, use positional embeddings
2. We set label_len == pred_len
3. We guess autoformer will be better used in auto-regression problem for decoder structure use input trend as hint.
4. Temporal Embeddings are usefull.
5. Moving Average are more usefull than pooling.


# Thanks

@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
