
# CAARMA: Class Augmentation with Adversarial Mixup Regularization

## Abstract

Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. Models must naturally generate embeddings that **cluster same-class instances compactly while maintaining separation across classes**.  
However, real-world speaker datasets often lack the **class diversity** required to generalize effectively.  

We introduce **CAARMA**, a class augmentation framework that:
- Generates **synthetic classes** via **adversarial mixup in the embedding space**  
- Employs an **adversarial refinement mechanism** to make synthetic classes indistinguishable from real ones  
- Expands the number of training classes, boosting zero-shot generalization  

Our experiments across multiple speaker verification benchmarks and zero-shot speech analysis tasks show **consistent gains with up to 8% improvement over strong baselines**.
<p align="center">

<div align=center>
	<img src=assets/data.png/>
</div>


---

## 🚀 Features
- 🔥 **Class Augmentation** using adversarial mixup regularization  
- 🧠 **Refinement Mechanism** ensures synthetic classes mimic real distributions  
- 🎯 Enhanced **zero-shot generalization** in speaker verification  
- 📈 Easy plug-in with popular SV backbones (ECAPA, MFA Conformer, Rawnet, etc.)  


---
## 📁 Directory Structure

```bash
caarma/
├── functions/                    # Dataset loaders (VoxCeleb, etc.)
│   ├── dataset.py
│   └── loader.py
├── helper/                # mixup 
│   ├── mixup_avg.py
├── models/                # Speaker embedding models
│   ├── MFA_Conformer.py
│   ├── ecapa_tdnn.py
│   ├── Raw_Net.py
│   ├── ska_tdnn.py
│   ├── discriminator_mix.py
│   └── build_model.py
├── configs/                  # YAML configs grouped by purpose
│   ├── base.yaml             # Original clean CAARMA training config
│   ├── base_diffusion_bridges2.yaml # Base diffusion-mixup Bridges-2 config
│   ├── gender_bridges2.yaml  # Gender diffusion-mixup Bridges-2 config
│   └── nationality_bridges2.yaml # Nationality mixup Bridges-2 config
├── train.py                # Training script
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Train your model


```bash
python train.py --config configs/base.yaml
```

Inside the selected config file, make sure to:
- Set the correct path to your **root**
- Set the correct path to your **trial_path**
- Set the correct path to your **dataset csv file**

For Bridges-2 runs, use the role-specific configs:

```bash
python mixup_gender.py --config configs/gender_bridges2.yaml --mode train --sl-mixup
python mixup_nationality.py --config configs/nationality_bridges2.yaml --mode train --nationality-mixup
python train.py --config configs/base_diffusion_bridges2.yaml
```

---
## 📌 Citation

If you find this useful in your research, please cite us:

```bibtex
@misc{CAARMA,
  title = {CAARMA: Class Augmentation with Adversarial Mixup Regularization},
  author = {Massa Baali and Xiang Li and Hao Chen and Syed Abdul Hannan and Rita Singh and Bhiksha Raj},
  year={2025},
  eprint={2503.16718},
  archivePrefix={arXiv},
  url={https://arxiv.org/pdf/2503.16718},
  primaryClass={cs.CL}
}
```
