# Deep Learning with Zero Initialization: Revisiting Symmetry Breaking and Gradient Flow

Jongwoo Seo (1st author)  
Preprint: [Link](https://www.researchsquare.com/article/rs-4890533/v2), [Link(the_latest_version)](https://www.researchsquare.com/article/rs-7341654/v1)  
LinkedIn: [Link](https://www.linkedin.com/in/jongwoo-seo/)  
Personal Website: [Link](https://sites.google.com/view/jongwooseo/)  
CV: [Link](https://sites.google.com/view/jongwooseo/cv?authuser=0)  
Google Scholar: [Link](https://scholar.google.co.kr/citations?hl=en&user=ikhaAuoAAAAJ)  
Github: https://github.com/sjw007s/Deep-Learning-with-Zero-Initialization-Revisiting-Symmetry-Breaking-and-Gradient-Flow  
Nationality: South Korean  
Contact: sjw007s@korea.ac.kr  

📢📢📢  
1. Seeking PhD positions and industry opportunities — email me.  

2. Understanding the gradient flow between operations during forward and backward propagation is not exclusive knowledge—it’s something anyone can learn. Still, some researchers may not be familiar with it. I'm open to contributing—whether it’s support, collaboration, or consulting—especially when proper credit is given (e.g., co-authorship, employment opportunity, or other fair recognition). If you're interested, feel free to reach out.

📢📢📢  

---

### 1. The four files below provide dataloader code for each dataset.

- `CNN_CIFAR_100_github.ipynb`  
- `CNN_CIFAR_10_github.ipynb`  
- `CNN_Mnist_github.ipynb`
- `tiny_imagenet.py`

---

### 2. The three files below provide code for each model.  
You can reproduce results by combining the previously mentioned dataloaders with each model.

- `cifar-10-Mixer_github.py`  
- `cifar-10-ViT_github.py`  
- `cifar-10-resnet_github.py`

---

### 3. All you need to edit is just 2 lines:

```python
torch.nn.init.zeros_(a.weight) #"a" may vary depending on the code.
torch.nn.init.zeros_(a.bias) #"a" may vary depending on the code.
```
---

### 4. Functions like train() and test() may differ slightly depending on data preprocessing methods and models.

### 5. These files contain MLP-Mixer models on the Tiny ImageNet dataset. There are no randomly initialized parameters at all.
- `tiny_imagenet_allconstant.py`
- `tiny_imagenet_all_zero_including_normalization_layer` (The accuracy is around 20%, so its performance is poor and not recommended.)  
