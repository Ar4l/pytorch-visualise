## pytorch-visualise
Visualise layer inputs &amp; outputs of Pytorch models. This is a bare-bones file, and while I don't intend on actively developing it, it may be useful to others. 

#### Usage
Wrap any Pytorch module, and specify sub-modules of interest. 

```py
model = AutoModel.from_pretrained('huggingface/CodeBERTa-small-v1')
Visualise(model, 'roberta.encoder', save_to='encoder_states.png')

model(**inputs) # generates colour-maps of hidden states in each submodule
```

#### Installation
I'm not publishing this, just copy over the `visualise.py` file.
