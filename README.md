## pytorch-visualise
Visualise layer inputs &amp; outputs of Pytorch models. This is a bare-bones file, and while I don't intend on actively developing it, it may be useful to others. 

#### Usage
Wrap any Pytorch module, and specify sub-modules of interest. 

```py
model = AutoModel.from_pretrained('huggingface/CodeBERTa-small-v1')
Visualise(model, ['roberta.encoder', 'roberta.pooler'], save_to='encoder_states.png')

model(**inputs) # generates colour-map of hidden states in each submodule
```

#### `Visualise` Arguments
`Visualise`, besides the model and a list of layers, takes four optional inputs:
- `print_weights = False`  whether to print weights to the console
- `max_weights = 100` number of weights to print (trims leading dims)
- `plot = False` whether to show the colour-map of the layer inputs and outputs (useful in notebooks)
- `save_to :str` where to save the colour-map (also useful in keeping your notebook uncluttered)

#### Installation
I'm not publishing this, just copy over the `visualise.py` file.
