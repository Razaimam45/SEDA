# On enhancement of SEVIT for Chest XRay Classification using Defensive Distillation and Adversarial Training. 

<hr />

![Method](Figures/Method.jpg)

## Abstract 
With the recent adaptation of deep learning in medical imaging problems, elevated vulnerabilities have been explored in reent CNN
and ViT based solutions. The vulnerability of ViTs to adversarial, privacy, and confidentiality attacks raise serious concerns about their reliability in medical settings. This work aims to enhance the robustness of existing benchmark solution based on self-ensembling ViTs in
tuberculosis chest X-ray classification problem. In the proposed work,
we have presented a novel SEVIT-CNN architecture built over SEVIT,
that utilizes the CNN modules for improved computational efficiency
and robustness utilizing the adversarial training and defensive distillation. The proposed approach leverages the fact that adversarial training
when performed with the combination of defensive distillation, presents
significantly higher robustness against adversaries. CNN’s efficiency in
learning spatial features through convolution operations at various levels
of abstraction, along with training the model with adversarial examples
improves its ability to handle perturbations and generalize better. By creating a distilled model with soft probabilities, uncertainty and variation
are introduced into the output probabilities, making it more difficult for
privacy attacks like model extraction. Extensive experiments performed
with the proposed architecture on publicly available Tuberculosis X-ray
dataset shows efficacy in terms of computational efficiency and enhanced
robustness.
## Keywords: Ensembling · Adversarial Attack · Defensive Distillation
