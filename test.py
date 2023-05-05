import argparse
import torch
from tqdm import tqdm
import os
from models import mlp
from data.dataset import data_loader


root_dir = "./data/TB_data"

def test_vit(model, dataloader_test):
    """
    This function used to test ViT. 

    Args: 
        model: ViT model
        dataaloader_test: loader for test images 
    
    return: 
        Avg test accuracy of ViT
    
    """
    test_acc = 0.0
    for images, labels in tqdm(dataloader_test): 
        images = images.cuda()
        labels= labels.cuda()
        with torch.no_grad(): 
            model.eval()
            output = model(images)
            prediction = torch.argmax(output, dim=-1)
            acc = sum(prediction == labels).float().item()/len(labels)
            test_acc += acc
    print(f'Testing accuracy = {(test_acc/len(dataloader_test)):.4f}')

    return round(test_acc/len(dataloader_test),2)

def test_mlps(mlps_list, data_loader, mlp_root_dir, model):
    length = (len(mlps_list))

    for index in range(length): 
        acc_avg = 0.0
        mlp_in = os.path.join(mlp_root_dir, mlps_list[index])
        # print(mlp_in)
        print(mlp_in, f": MLP at index {index} has been loaded")
        classifier = torch.load(mlp_in).cuda()
        classifier.eval()
        data_loader_test = data_loader['test']

        with torch.no_grad(): 
            for image, label in tqdm(data_loader_test):
                image = image.cuda()
                label = label.cuda()
                x = model.patch_embed(image)
                x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = model.pos_drop(x + model.pos_embed)
                for i in range(index+1):
                    x = model.blocks[i](x)
                output = classifier(x[:,1:,:])
                predictions = torch.argmax(output, dim=-1)
                acc = torch.sum(predictions == label).item()/len(label)
                acc_avg += acc
            print(f'Accuracy of block {index} = {(acc_avg/len(data_loader_test)):.3f}')
        print(f'================= Block {index} Finished ==============\n\n')
            
    pass

parser = argparse.ArgumentParser(description='Testing ViT or MLPs')

parser.add_argument('--model_name', type=str , choices=['ViT','MLPs'],
                    help='Choose between ViT or MLPs')
parser.add_argument('--vit_path', type=str ,
                    help='pass the path of downloaded ViT')
parser.add_argument('--mlp_path', type=str ,
                    help='pass the path for the downloaded MLPs folder')
args = parser.parse_args()

loader_, dataset_ = data_loader(root_dir=root_dir)

model = torch.load(args.vit_path).cuda()
model.eval()

if args.model_name == 'ViT':
    acc = test_vit(model=model, dataloader_test=loader_['test'])
else:
    mlps_list = sorted(os.listdir(args.mlp_path))
    print(mlps_list)
    acc = test_mlps(mlps_list= mlps_list, data_loader=loader_, mlp_root_dir=args.mlp_path, model=model)

