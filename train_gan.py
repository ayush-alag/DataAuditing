import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import torch.nn as nn
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# We train the discriminator by predicting 1 for real and 0 for fake (generated)
def train_discriminator(images, batch_size, latent_size):
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
        
    # Loss for real images
    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Combine losses
    d_loss = d_loss_real + d_loss_fake
    # Reset gradients
    reset_grad()
    # Compute gradients
    d_loss.backward()
    # Adjust the parameters using backprop
    d_optimizer.step()
    
    return d_loss, real_score, fake_score

def save_fake_images(index, sample_vectors):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)

    # save png images
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=50)

    # save tensors
    fake_ptname = 'fake_images-{0:0=4d}.pt'.format(index)
    torch.save(denorm(fake_images), os.path.join(sample_dir, fake_ptname))

# We train the generator by trying to get the discriminator to predict 1
# on the fake images
def train_generator(batch_size, latent_size):
    # Generate fake images and calculate loss
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(D(fake_images), labels)

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images

# k, resizing, converting to image, and labeling in calibration module
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--epoch', metavar='E', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', metavar='B', type=int, default=100,
                        help='batch size')
    parser.add_argument('--expt', type=str, default="")
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--train_size', type=int, default=60000)
    parser.add_argument('--num_generate', type=int, default=10000)

    args = parser.parse_args()
    
    if args.dataset != "MNIST":
        print("Error: unsupported dataset")
        exit

    mnist = MNIST(root='./data', 
              train=True, 
              download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
    
    print(len(mnist)) # should be 60,000
    
    mnist_train = torch.utils.data.Subset(mnist, list(range(0, args.train_size)))
    dataloader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=16, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define simple generator/discriminator models
    D = nn.Sequential(
        nn.Linear(args.image_size * args.image_size, args.hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(args.hidden_size, 1),
        nn.Sigmoid())
    G = nn.Sequential(
        nn.Linear(args.latent_size, args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, args.image_size * args.image_size),
        nn.Tanh())
    
    D.to(device)
    G.to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

    # TODO: add to the saves_new/args.expt folder
    sample_dir = f'saves_new/{args.expt}/{args.dataset}/cal_gan/samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    ckpt_dir = f'saves_new/{args.expt}/{args.dataset}/cal_gan/checkpoints'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Save some real images
    for images, _ in dataloader:
        images = images.reshape(images.size(0), 1, args.image_size, args.image_size)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=10)
        break
    
    sample_vectors = torch.randn(args.num_generate, args.latent_size).to(device)
        
    # Before training
    save_fake_images(0, sample_vectors)

    total_step = len(dataloader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(args.epoch):
        for i, (images, _) in enumerate(dataloader):
            # Load a batch & transform to vectors
            images = images.reshape(args.batch_size, -1).to(device)
            
            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images, args.batch_size, args.latent_size)
            g_loss, fake_images = train_generator(args.batch_size, args.latent_size)
            
            # Inspect the losses
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, args.epoch, i+1, total_step, d_loss.item(), g_loss.item(), 
                            real_score.mean().item(), fake_score.mean().item()))
            
        if (epoch + 1) % 5 == 0:
            # Sample and save images
            save_fake_images(epoch+1, sample_vectors)
        
        if epoch % 50 == 0:
            torch.save(G.state_dict(), f'{ckpt_dir}/G_epoch_{epoch}.ckpt')
            torch.save(D.state_dict(), f'{ckpt_dir}/D_epoch_{epoch}.ckpt')
            
    # Save the model checkpoints 
    torch.save(G.state_dict(), f'{ckpt_dir}/G.ckpt')
    torch.save(D.state_dict(), f'{ckpt_dir}/.ckpt')