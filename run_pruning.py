import argparse
import torch
import numpy as np
import os

from torchvision import models
import torch.nn.utils.prune as prune
from arch import LocationModel, MLP
from data_utils import LocationDataModule, COVIDxDataModule, MNISTDataModule, MNISTLeNetModule
from trainer import append_dropout

def calculate_accuracy(model, dataloader, device):
   correct = 0
   with torch.no_grad():
      for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            if output.data.shape[1] == 1:
               pred = np.where(np.asarray(output.data.detach().cpu()) >= 0.5, 1, 0)
               correct += np.sum(pred == labels.detach().cpu().numpy().astype(int))
            else:
               pred = output.data.max(1)[1]
               correct += pred.eq(labels.view(-1)).sum().item()

   correct /= len(dataloader.dataset)

   return correct

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition maybe?

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Run an experiment.')
   parser.add_argument('--dataset', type=str, default="Location", help='dataset')
   parser.add_argument('--expt', type=str, default="", help='experiment name')
   parser.add_argument('--method', type=str, default="mag", help="pruning method")
   parser.add_argument('--sparsity', type=float, default=1, help="amount to compress")
   parser.add_argument('--mode', type=str, default="base", help="base or cal")
   parser.add_argument('--k', type=int, default=0, help="k for calibration model")
   parser.add_argument('--size', type=int, default=10000, help="size for calibration model")
   parser.add_argument('--dropout', type=float, default=0.0, help="amount of dropout")
   parser.add_argument('--dim', type=int, default=256, help='hidden dim of MLP')
   parser.add_argument('--epochs', type=int, default=50, help="50 or 200 training")
   parser.add_argument('--lenet', type=t_or_f, default=False)

   args = parser.parse_args()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Load the target model
   model = None
   parameters_to_prune = None
   if args.dataset == 'Location':
      dataset = LocationDataModule(mode='cal', k=args.k, calset=args.dataset)
      model = LocationModel.LocationMLP(dataset.input_shape, dataset.output_dims, dropout_probability=args.dropout).to(device)
      parameters_to_prune = (
         (model.first_block[0], 'weight'),
         (model.second_block[0], 'weight'),
         (model.third_block[0], 'weight'),
         (model.fourth_block[0], 'weight'),
         (model.final_block[0], 'weight'),
      )
   elif args.dataset == "MNIST":
      if not args.lenet:
         model = MLP.MLP(28, args.dim, 10, args.dropout).to(device)
         parameters_to_prune = (
            (model.fc1, 'weight'),
            (model.fc2, 'weight'),
            (model.fc3, 'weight'),
         )
      else:
         model = MLP.LeNet5(10, args.dropout).to(device)
         parameters_to_prune = (
            (model.features[0], 'weight'),
            (model.features[3], 'weight'),
            (model.classifier[0], 'weight'),
            (model.classifier[3], 'weight'),
            (model.classifier[6], 'weight')
         )

   elif args.dataset == "COVIDx":
      # no dropout for covid model
      model = models.resnet18(pretrained=False, num_classes=2).to(device)
      parameters_to_prune = []
      for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
               parameters_to_prune.append((module, "weight"))
      parameters_to_prune = tuple(parameters_to_prune)
      append_dropout(model, args.dropout)

   
   prefix = ""
   if args.mode == "cal":
      prefix = "caldata=" + args.dataset + "_k=" + args.k + "_size=" + args.size

   model_path = f'saves_new/{args.expt}/{args.dataset}/{args.mode}/{prefix}training_epoch{args.epochs}.pkl'
   model_dict = torch.load(model_path)
   model.load_state_dict(model_dict)

   # get training accuracy before the model pruning
   model.eval()
   if args.dataset == "Location":
      eval_data = LocationDataModule(mode='base', k=args.k, calset=args.dataset)
   elif args.dataset == "MNIST":
      if not args.lenet:
         eval_data = MNISTDataModule(mode='base', k=args.k, calset=args.dataset)
      else:
         eval_data = MNISTLeNetModule(mode='base', k=args.k, calset=args.dataset)
   elif args.dataset == "COVIDx":
      eval_data = COVIDxDataModule(mode="base", k=args.k, calset=args.dataset)

   pre_train = calculate_accuracy(model, eval_data.train_dataloader(), device)
   pre_test = calculate_accuracy(model, eval_data.test_dataloader(), device)
   print("Pre-pruning training accuracy:", pre_train)
   print("Pre-pruning testing accuracy", pre_test)

   # prune the target model with magnitude pruning
   prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=args.sparsity,
   )

   if args.dataset == "Location":
      print(
         "Sparsity in first_linear.weight: {:.2f}%".format(
            100. * float(torch.sum(model.first_block[0].weight == 0))
            / float(model.first_block[0].weight.nelement())
         )
      )
      print(
         "Sparsity in second_linear.weight: {:.2f}%".format(
            100. * float(torch.sum(model.second_block[0].weight == 0))
            / float(model.second_block[0].weight.nelement())
         )
      )
      print(
         "Sparsity in third_linear.weight: {:.2f}%".format(
            100. * float(torch.sum(model.third_block[0].weight == 0))
            / float(model.third_block[0].weight.nelement())
         )
      )
      print(
         "Sparsity in fourth_linear.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fourth_block[0].weight == 0))
            / float(model.fourth_block[0].weight.nelement())
         )
      )
      print(
         "Sparsity in fifth_linear.weight: {:.2f}%".format(
            100. * float(torch.sum(model.final_block[0].weight == 0))
            / float(model.final_block[0].weight.nelement())
         )
      )
      print(
         "Global sparsity: {:.2f}%".format(
            100. * float(
                  torch.sum(model.first_block[0].weight == 0)
                  + torch.sum(model.second_block[0].weight == 0)
                  + torch.sum(model.third_block[0].weight == 0)
                  + torch.sum(model.fourth_block[0].weight == 0)
                  + torch.sum(model.final_block[0].weight == 0)
            )
            / float(
                  model.first_block[0].weight.nelement()
                  + model.second_block[0].weight.nelement()
                  + model.third_block[0].weight.nelement()
                  + model.fourth_block[0].weight.nelement()
                  + model.final_block[0].weight.nelement()
            )
         )
      )

   elif args.dataset == "MNIST":
      if not args.lenet:
         print(
            "Sparsity in fc1.weight: {:.2f}%".format(
               100. * float(torch.sum(model.fc1.weight == 0))
               / float(model.fc1.weight.nelement())
            )
         )
         print(
            "Sparsity in fc2.weight: {:.2f}%".format(
               100. * float(torch.sum(model.fc2.weight == 0))
               / float(model.fc2.weight.nelement())
            )
         )
         print(
            "Sparsity in fc3.weight: {:.2f}%".format(
               100. * float(torch.sum(model.fc3.weight == 0))
               / float(model.fc3.weight.nelement())
            )
         )
         print(
            "Global sparsity: {:.2f}%".format(
               100. * float(
                     torch.sum(model.fc1.weight == 0)
                     + torch.sum(model.fc2.weight == 0)
                     + torch.sum(model.fc3.weight == 0)
               )
               / float(
                     model.fc1.weight.nelement()
                     + model.fc2.weight.nelement()
                     + model.fc3.weight.nelement()
               )
            )
         )
      else:
         print(
            "Sparsity in first_conv.weight: {:.2f}%".format(
               100. * float(torch.sum(model.features[0].weight == 0))
               / float(model.features[0].weight.nelement())
            )
         )
         print(
            "Sparsity in second_conv.weight: {:.2f}%".format(
               100. * float(torch.sum(model.features[3].weight == 0))
               / float(model.features[3].weight.nelement())
            )
         )
         print(
            "Sparsity in first_linear.weight: {:.2f}%".format(
               100. * float(torch.sum(model.classifier[0].weight == 0))
               / float(model.classifier[0].weight.nelement())
            )
         )
         print(
            "Sparsity in second_linear.weight: {:.2f}%".format(
               100. * float(torch.sum(model.classifier[3].weight == 0))
               / float(model.classifier[3].weight.nelement())
            )
         )
         print(
            "Sparsity in third_linear.weight: {:.2f}%".format(
               100. * float(torch.sum(model.classifier[6].weight == 0))
               / float(model.classifier[6].weight.nelement())
            )
         )
         print(
            "Global sparsity: {:.2f}%".format(
               100. * float(
                     torch.sum(model.features[0].weight == 0)
                     + torch.sum(model.features[3].weight == 0)
                     + torch.sum(model.classifier[0].weight == 0)
                     + torch.sum(model.classifier[3].weight == 0)
                     + torch.sum(model.classifier[6].weight == 0)
               )
               / float(
                     model.features[0].weight.nelement()
                     + model.features[3].weight.nelement()
                     + model.classifier[0].weight.nelement()
                     + model.classifier[3].weight.nelement()
                     + model.classifier[6].weight.nelement()
               )
            )
         )
   elif args.dataset == "COVIDx":
      for i, (module_name, module) in enumerate(model.named_modules()):
         if isinstance(module, torch.nn.Conv2d):
            print(
            "Sparsity in Layer {0:3d}: {1:.2f}%".format(i, 
               100. * float(torch.sum(module.weight == 0))
               / float(module.weight.nelement())
            )
      )

   # evaluate its accuracy on training??
   post_train = calculate_accuracy(model, eval_data.train_dataloader(), device)
   post_test = calculate_accuracy(model, eval_data.test_dataloader(), device)
   print("Post-pruning training accuracy:", post_train)
   print("Post-pruning testing accuracy", post_test)

   # override the target model so we can run EMA
   print("Saving the model...")
   print(model_path)

   for (parameter, _) in parameters_to_prune:
      prune.remove(parameter, name="weight")
      
   torch.save(model.state_dict(), model_path)