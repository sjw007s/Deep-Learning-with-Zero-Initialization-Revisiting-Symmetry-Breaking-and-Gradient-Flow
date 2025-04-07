sjw007s@korea.ac.kr  
https://www.researchsquare.com/article/rs-4890533/v1

---

### 1. The three files below provide the dataloader code for each dataset.

- `CNN_CIFAR_100_github.ipynb`  
- `CNN_CIFAR_10_github.ipynb`  
- `CNN_Mnist_github.ipynb`

---

### 2. The three files below provide code for each model.  
You can reproduce the results by combining the previously mentioned dataloaders with each model.

- `cifar-10-Mixer_github.py`  
- `cifar-10-ViT_github.py`  
- `cifar-10-resnet_github.py`

---

### 3. All you need to edit is just 2 lines:

```python
torch.nn.init.zeros_(a.weight) #"a" may vary depending on the code.
torch.nn.init.zeros_(a.bias) #"a" may vary depending on the code.
