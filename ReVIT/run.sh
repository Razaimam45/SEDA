## Test ViT and Intermediate MLPs 
To test the performance of the ViT on the test samples, run the following commmand: 

```
python test.py --model_name ViT --vit_path  'models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth'
````

To test the intermediate MLPs performance on the test samples, run the following command: 

```
python test.py --model_name MLPs  --mlp_path  'models/MLP_new_chest' --vit_path 'models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth'
```


## Generate Adversarial Attacks from Test Samples
To generate attacks from clean samples and ViT as a target model, run the the following command: 

```
python generate_attacks.py --epsilons "0.03" --attack_list "FGSM" "PGD" "BIM" "AUTOPGD" "CW" --vit_path 'models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth' --attack_images_dir '/home/raza.imam/Documents/HC701B/Project/attack_images2'
```

## Adversarial Robustness (Majority Voting)
In order to evaluate **SEViT** on **Clean samples:** 

```
python majority_voting.py --images_type clean --image_folder_path 'data/TB_data' --vit_path 'models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth' --mlp_path 'models/MLP_new_chest'
```

To evaluate **SEViT** performance on **Attack samples:**

```
python majority_voting.py --images_type adversarial --attack_name 'AUTOPGD' --image_folder_path 'attack_images' --vit_path 'models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth' --mlp_path 'models/MLP_new_chest'
```

## Adversarial Detection
To perform adversarial detection and save the ROC figure for the generated adversarial samples, run the following command: 

```
python adversarial_detection.py --attack_name 'AUTOPGD' --clean_image_folder_path 'data/TB_data' --attack_image_folder_path 'attack_images' --vit_path 'models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth' --mlp_path 'models/MLP_new_chest'
```