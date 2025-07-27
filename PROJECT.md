# Goal:
Replicate the paper 'grokking modular addition'

## Paper Method
- Input (a b =) where a,b are one-hot encoded vectors of size P (P=113)
- One layer ReLU transformer
- Token embed dimension d=128
- Learned positional embeddings
- Four attention heads of d/4=32
- MLP dimension of 512
- No layernorm/tying of embed/unembed matrices
- Adam with gamma=0.001, weight decay lambda=1
- 40k epochs
- Uses 30% of inputs for training (0.3 * P**2)
- Grokking requires regularization
