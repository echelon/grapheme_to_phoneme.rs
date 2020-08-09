# Reference

These are the original sources for training (unconfirmed) and evaluation in case
they go offline in the future.

Training (I haven't done this myself): 

- https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb
- No stated license; included as a reference only.

Evaluation (which this crate is based opon):

- https://github.com/Kyubyong/g2p
- [Apache 2.0 LICENSE](https://github.com/Kyubyong/g2p/blob/master/LICENSE.txt)

Model

- https://github.com/Kyubyong/g2p/blob/master/g2p_en/checkpoint20.npz
- This would not parse with the `ndarray-npy` crate, so I had to unzip and rebundle it
  using ndarray-npy's npz writer. The original is provided as reference in the event
  the repository moves or goes offline.

