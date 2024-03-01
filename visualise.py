from torch import nn
import torch, os, pdb
import matplotlib.pyplot as plt

def rec_getmodule(module: nn.Module) -> list[nn.Module]:
    ''' Recursively get all submodules of a module '''
    if len([c for c in module.children()]) == 0:
        return [module]
    modules = []
    for submodule in module.children():
        modules += rec_getmodule(submodule)
    return modules

WELCOME_STR = ''' 
    \033[1mPytorch Visualiser\033[0m by Aral
    Visualising {}, consisting of {} 
    {} '''

class Visualise:
    def __init__(self, model: nn.Module, module_names:list[str], 
                 print_weights=False, max_weights=100, plot=False, save_to:str=None):
        ''' Visualise the weights, and hidden layers of modules in the model.
            - model is the instantiated model to watch 
            - module_names specifies which modules (and its children) to watch 
            
            - `print_weights` prints weights, biases, and inputs of each layer 
            - `max_weights` specifies the maximum number of weights to print
            - `plot` specifies whether to plot the hidden states of each layer
            - `save_to` is the path to save the plot to '''

        self.model = model
        self.print_weights = print_weights
        self.max_weights = max_weights

        self.save_to = save_to 
        self.vis = plot if save_to is None else True

        modules = [model.get_submodule(module_name) for module_name in module_names]
        modules = {module: rec_getmodule(module) for module in modules}
        self.modules = modules

        self.n_mod = len(modules)
        self.n_sub = sum([len(submodules) for submodules in modules.values()])

        for module, submodules in modules.items():
            # print the hidden states at the start of this module
            # and create a plot for subsequent submodules to plot on
            module.register_forward_pre_hook(self._first_hook)

            for submodule in submodules[:-1]:
                # print the hidden state after it's been modified by each submodule
                # and put it on a subplot from the first hook
                submodule.register_forward_hook(self._middle_hook)

            # show the entire plot 
            last_submodule = submodules[-1]
            last_submodule.register_forward_hook(self._last_hook)

        print(WELCOME_STR.format(
            f'{self.n_mod} module{"s" if self.n_mod > 1 else ""}', 
            f'{self.n_sub} submodule{"s" if self.n_sub > 1 else ""} in total', 
            f'Plotting to {save_to}\n' if save_to is not None else '')
        )
              
        for module, submodules in modules.items():
            print(f'\t\033[1m{module.__class__.__name__}\033[0m')
            for submodule in submodules:
                print(f'\t  {submodule}')

    def _first_hook(self, module: nn.Module, inputs: tuple[torch.tensor]):
        ''' Visualise the inputs to the module '''
        label = module.__class__.__name__
        print(f'\033[1m{label}\033[0m {inputs[0].shape}')

        if self.vis:
            # create a plot with self.n_sub subplots
            self.fig, self.ax = plt.subplots(self.n_mod + 2*self.n_sub, 1, figsize=(5, 5*(self.n_sub+1)))
            self.ax = self.ax.flatten()
            self.i = 0
            self.plot(inputs[0], f'{label} input')
        self.dprint(inputs[0])

    def _middle_hook(self, layer: nn.Module, *hidden_states: tuple[torch.Tensor], **kwargs):
        ''' Visualise the current hidden_states, and any potential inputs '''

        inputs, outputs = *hidden_states[0], hidden_states[1]

        if self.vis and len([c for layer in layer.children()]) == 0:
            self.plot(inputs, f'{layer.__class__.__name__} input')
            self.plot(outputs, layer.__class__.__name__)

        print(f'\n\033[1m{layer}\033[0m {layer.shape if hasattr(layer, "shape") else ""}')
        if hasattr(layer, 'weight'):
            self.dprint(layer.weight)
        if hasattr(layer, 'bias'):
            self.dprint(layer.bias)

        if len([c for c in layer.children()]) == 0: # has no children, thus modifies hidden_states
            print(f'\033[1mhidden_states\033[0m {outputs.shape}')
            self.dprint(outputs)

    def _last_hook(self, module: nn.Module, inputs: tuple[torch.Tensor], *args, **kwargs):
        self._middle_hook(module, inputs, *args, **kwargs)

        if self.save_to is not None:
            plt.savefig(self.save_to)
            # do not show the image 
            plt.close()
        else: 
            plt.show()

    def dprint(self, tensor: torch.tensor, trunc = False):
        if not self.print_weights: return
        if tensor.shape.numel() > self.max_weights:

            shape = tensor.shape
            if shape[0] == 1 and len(shape) > 1:
                self.dprint(tensor[0], trunc=True)
            else:
                self.dprint(tensor[:shape[0]-1], trunc=True)

        else:
            if not trunc: 
                print(tensor)
            else:
                print(f'truncated to {tensor.shape}\n{tensor}')

    def plot(self, tensor: torch.tensor, label=''):

        og_shape = ','.join(str(s) for s in tensor.shape)
        # remove the batch dimension
        tensor = tensor.clone().detach().cpu()[0]
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        # put the larger dimension last always
        if tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.T

        new_shape = ','.join(str(s) for s in tensor.shape)
        label = f'{label} ({og_shape}) -> ({new_shape})'

        # plot the tensor as image,
        cax = self.ax[self.i].matshow(tensor.numpy())
        # cax.set_clim(tensor.min(), tensor.max())
        # set colorbar labels to include the min and max 
        # self.fig.colorbar(cax, ax=self.ax[self.i], orientation='horizontal')
        self.fig.colorbar(cax, ax=self.ax[self.i], orientation='horizontal', ticks=[tensor.min(), tensor.max()])

        self.ax[self.i].set_title(label)

        # if first dimension is 1, then set the y label to 1 
        if tensor.shape[0] == 1:
            self.ax[self.i].set_yticks([0])
            self.ax[self.i].set_yticklabels([1])
        
        self.i += 1