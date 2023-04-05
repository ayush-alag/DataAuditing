import os
import ast
import configparser

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset, TensorDataset
from torch.utils.data.dataloader import DataLoader

from config import *
from datasets.covidxdataset import COVIDxDataset
from datasets.cxrdataset import init_CXR

from arch import LocationModel


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


transform_svhn = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_lenet = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_lenet_rotate = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_lenet_noise = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AddGaussianNoise(0., 3.),
])


transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnist_rotate = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnist_noise = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AddGaussianNoise(0., 3.)
])

transform_usps = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


DEFAULT_DATA_DIR = './data'
DEFAULT_NUM_WORKERS = 16

# need all data to have 32 x 32 x 3 rather than 28 x 28 x 1
class MNISTLeNetModule():
    def __init__(self, batch_size: int = 32, data_dir: str = DEFAULT_DATA_DIR, num_workers: int = DEFAULT_NUM_WORKERS, k: float = 0, mode: str = 'base', calset: str = 'MNIST', use_own: bool = False, fold: int = 0):
        self.dir = data_dir
        self.data_dir = os.path.join(self.dir, 'mnist')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10

        self.k = k
        self.calset = calset
        self.use_own = use_own
        self.fold = fold
        self.mode = mode
        self.setup()
    
    def setup(self):
        # Assign train, test, cal, cal_test datasets for use in dataloaders
        if self.mode == 'base':
            trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                                  download=True, transform=transform_lenet)
            self.train_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['MNIST']['TRAINING_INDEX_START'], CONFIG['MNIST']['TRAINING_INDEX_END'])))
            self.test_set = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_lenet)

        elif self.mode == 'query' or (self.mode == 'cal' and self.use_own):
            trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                                  download=True, transform=transform_lenet)
            if self.fold > 0 and self.fold <= CONFIG['MNIST']['FOLD_NUM']:
                self.train_set = torch.utils.data.Subset(
                    trainset, list(range((self.fold-1)*2000, self.fold*2000)))
            elif self.fold == 0:        # SVHN
                trainset_svhn = torchvision.datasets.SVHN(
                    root='./data/svhn', split='train', download=True, transform=transform_lenet)
                self.train_set = torch.utils.data.Subset(
                    trainset_svhn, list(range(0, 2000)))
            else:                       # Unincluded fold in training
                self.train_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['MNIST']['TRAINING_OUT_INDEX_START'], CONFIG['MNIST']['TRAINING_OUT_INDEX_END'])))

            self.test_set = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_lenet)

        elif self.mode == 'cal':
            if self.calset == 'MNIST':
                trainset_ori = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                          download=True, transform=transform_lenet)
                trainset_rotate = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                             download=True, transform=transform_lenet_rotate)
                trainset_noise = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                            download=True, transform=transform_lenet_noise)

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_rotate = Subset(trainset_rotate, list(
                    range(int((100-self.k)/100*len(trainset_rotate)), int((100-self.k//2)/100*len(trainset_rotate)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k//2)/100*len(trainset_noise)), len(trainset_noise))))
                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_rotate, trainset_noise])

                self.test_set = torchvision.datasets.MNIST(
                    root=self.data_dir, train=True, download=True, transform=transform_lenet)
                self.test_set = Subset(self.test_set, list(
                    range(0, len(self.train_set))))     # part of the train set
            elif self.calset == 'FMNIST':
                trainset_ori = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                 download=True, transform=transform_lenet)
                trainset_rotate = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                    download=True, transform=transform_lenet_rotate)
                trainset_noise = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                   download=True, transform=transform_lenet_noise)

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_rotate = Subset(trainset_rotate, list(
                    range(int((100-self.k)/100*len(trainset_rotate)), int((100-self.k//2)/100*len(trainset_rotate)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k//2)/100*len(trainset_noise)), len(trainset_noise))))
                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_rotate, trainset_noise])

                self.test_set = torchvision.datasets.FashionMNIST(
                    root=self.data_dir, train=True, download=True, transform=transform_lenet)
                self.test_set = Subset(self.test_set, list(
                    range(0, len(self.train_set))))     # part of the train set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)



class MNISTDataModule():
    def __init__(self, batch_size: int = 32, data_dir: str = DEFAULT_DATA_DIR, num_workers: int = DEFAULT_NUM_WORKERS, k: float = 0, mode: str = 'base', calset: str = 'MNIST', use_own: bool = False, fold: int = 0):
        self.dir = data_dir
        self.data_dir = os.path.join(self.dir, 'mnist')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10

        self.k = k
        self.calset = calset
        self.use_own = use_own
        self.fold = fold
        self.mode = mode
        self.setup()

    def setup(self):
        # Assign train, test, cal, cal_test datasets for use in dataloaders
        if self.mode == 'base':
            trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                                  download=True, transform=transform_mnist)
            self.train_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['MNIST']['TRAINING_INDEX_START'], CONFIG['MNIST']['TRAINING_INDEX_END'])))
            self.test_set = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_mnist)

        elif self.mode == 'query' or (self.mode == 'cal' and self.use_own):
            trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                                  download=True, transform=transform_mnist)
            if self.fold > 0 and self.fold <= CONFIG['MNIST']['FOLD_NUM']:
                self.train_set = torch.utils.data.Subset(
                    trainset, list(range((self.fold-1)*2000, self.fold*2000)))
            elif self.fold == 0:        # SVHN
                trainset_svhn = torchvision.datasets.SVHN(
                    root='./data/svhn', split='train', download=True, transform=transform_svhn)
                self.train_set = torch.utils.data.Subset(
                    trainset_svhn, list(range(0, 2000)))
            else:                       # Unincluded fold in training
                self.train_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['MNIST']['TRAINING_OUT_INDEX_START'], CONFIG['MNIST']['TRAINING_OUT_INDEX_END'])))

            self.test_set = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_mnist)

        elif self.mode == 'cal':
            if self.calset == 'MNIST':
                trainset_ori = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                          download=True, transform=transform_mnist)
                trainset_rotate = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                             download=True, transform=transform_mnist_rotate)
                trainset_noise = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                            download=True, transform=transform_mnist_noise)

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_rotate = Subset(trainset_rotate, list(
                    range(int((100-self.k)/100*len(trainset_rotate)), int((100-self.k//2)/100*len(trainset_rotate)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k//2)/100*len(trainset_noise)), len(trainset_noise))))
                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_rotate, trainset_noise])

                self.test_set = torchvision.datasets.MNIST(
                    root=self.data_dir, train=True, download=True, transform=transform_mnist)
                self.test_set = Subset(self.test_set, list(
                    range(0, len(self.train_set))))     # part of the train set
            elif self.calset == 'FMNIST':
                trainset_ori = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                 download=True, transform=transform_mnist)
                trainset_rotate = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                    download=True, transform=transform_mnist_rotate)
                trainset_noise = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                   download=True, transform=transform_mnist_noise)

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_rotate = Subset(trainset_rotate, list(
                    range(int((100-self.k)/100*len(trainset_rotate)), int((100-self.k//2)/100*len(trainset_rotate)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k//2)/100*len(trainset_noise)), len(trainset_noise))))
                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_rotate, trainset_noise])

                self.test_set = torchvision.datasets.FashionMNIST(
                    root=self.data_dir, train=True, download=True, transform=transform_mnist)
                self.test_set = Subset(self.test_set, list(
                    range(0, len(self.train_set))))     # part of the train set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class COVIDxDataModule():
    def __init__(self, batch_size: int = 32, data_dir: str = DEFAULT_DATA_DIR, num_workers: int = DEFAULT_NUM_WORKERS, k: float = 0, mode: str = 'base', calset: str = 'COVID', use_own: bool = False, fold: int = 0):
        self.dir = data_dir
        self.data_dir = os.path.join(self.dir, 'covid')
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.k = k
        self.calset = calset
        self.use_own = use_own
        self.fold = fold
        self.mode = mode
        self.setup()

    def setup(self):
        if self.mode == 'base':
            trainset = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                     dim=(224, 224))
            self.train_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['COVID']['TRAINING_INDEX_START'], CONFIG['COVID']['TRAINING_INDEX_END'])))
            self.test_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['COVID']['TRAINING_TEST_INDEX_START'], CONFIG['COVID']['TRAINING_TEST_INDEX_END'])))
            print(len(self.train_set))

        elif self.mode == 'query' or (self.mode == 'cal' and self.use_own):
            trainset = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                     dim=(224, 224))
            if self.fold > 0 and self.fold <= CONFIG['COVID']['FOLD_NUM']:
                print((self.fold-1)*800, self.fold*800)
                self.train_set = torch.utils.data.Subset(
                    trainset, list(range((self.fold-1)*800, self.fold*800)))
            elif self.fold == 0:        # SVHN, TODO...
                trainset_RSNA = init_CXR(mode='test')
                rand_idx = np.random.randint(0, len(trainset_RSNA), 800)
                self.train_set = torch.utils.data.Subset(
                    trainset_RSNA, rand_idx)
            else:                       # Unincluded fold in training
                self.train_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['COVID']['TRAINING_OUT_INDEX_START'], CONFIG['COVID']['TRAINING_OUT_INDEX_END'])))

            self.test_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['COVID']['CAL_TEST_INDEX_START'], CONFIG['COVID']['CAL_TEST_INDEX_END'])))

        elif self.mode == 'cal':
            if self.calset == 'COVIDx':
                trainset = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                         dim=(224, 224))
                trainset_ori = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['COVID']['CAL_TRAIN_INDEX_START'], CONFIG['COVID']['CAL_TRAIN_INDEX_END'])))

                trainset_noise = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                               dim=(224, 224), noise=True)
                trainset_noise = torch.utils.data.Subset(trainset_noise, list(range(
                    CONFIG['COVID']['CAL_TRAIN_INDEX_START'], CONFIG['COVID']['CAL_TRAIN_INDEX_END'])))

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k)/100*len(trainset_noise)), len(trainset_noise))))

                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_noise])

                self.test_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['COVID']['CAL_TEST_INDEX_START'], CONFIG['COVID']['CAL_TEST_INDEX_END'])))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)



class LocationDataModule():
    def __init__(self, num_workers: int = DEFAULT_NUM_WORKERS, k: float = 0, mode: str = 'base', calset: str = 'Location', use_own: bool = False, fold: int = 0, expt=""):
        self.dataset = 'location'

        config = configparser.ConfigParser()
        config.read('MemGuard/config.ini')

        self.config = config
        print("initializing from config..")

        self.batch_size = int(self.config[self.dataset]["batch_size"])
        self.num_workers = num_workers

        self.k = k
        self.calset = calset
        self.use_own = use_own
        self.fold = fold
        self.mode = mode
        self.expt = expt

        self.output_dims = int(self.config[self.dataset]["num_classes"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_filepath=str(self.config[self.dataset]['all_data_path'])
        self.index_filepath=str(self.config[self.dataset]['shuffle_index'])

        self.user_training_data_range=self.config[self.dataset]['user_training_data_index_range']
        self.user_training_data_range=ast.literal_eval(self.user_training_data_range)

        self.user_testing_data_range=self.config[self.dataset]['user_testing_data_index_range']
        self.user_testing_data_range=ast.literal_eval(self.user_testing_data_range)

        self.defense_member_data_index_range=self.config[self.dataset]['defense_member_data_index_range']
        self.defense_member_data_index_range=ast.literal_eval(self.defense_member_data_index_range)

        self.defense_nonmember_data_index_range=self.config[self.dataset]['defense_nonmember_data_index_range']
        self.defense_nonmember_data_index_range=ast.literal_eval(self.defense_nonmember_data_index_range)


        self.attacker_train_member_data_range=self.config[self.dataset]['attacker_train_member_data_range']
        self.attacker_train_member_data_range=ast.literal_eval(self.attacker_train_member_data_range)

        self.attacker_train_nonmember_data_range=self.config[self.dataset]['attacker_train_nonmember_data_range']
        self.attacker_train_nonmember_data_range=ast.literal_eval(self.attacker_train_nonmember_data_range)

        self.attacker_evaluate_member_data_range=self.config[self.dataset]['attacker_evaluate_member_data_range']
        self.attacker_evaluate_member_data_range=ast.literal_eval(self.attacker_evaluate_member_data_range)

        self.attacker_evaluate_nonmember_data_range=self.config[self.dataset]['attacker_evaluate_nonmember_data_range']
        self.attacker_evaluate_nonmember_data_range=ast.literal_eval(self.attacker_evaluate_nonmember_data_range)

        self.attacker_nonmember_even=self.config[self.dataset]['attacker_nonmember_even']
        self.attacker_nonmember_even=ast.literal_eval(self.attacker_nonmember_even)

        (x_train,_), _ = self.input_data_user()
        self.input_shape = x_train.shape[1:]

        print("setting up..")
        self.setup()
    
    def clean_y(self, y):
        y = torch.tensor(y, dtype=torch.long)
        return y


    # set self.train_set and self.test_set
    # base is the base model: get the data from the helper methods and set the variables
    # query is how we generate our stats: one should be svhn, others should be from the datasetP
    # calibration is to train the calibration models with different values of k: need to add a transform for location
    def setup(self):
        if self.mode == "base":
            (x_train,y_train),(x_test,y_test) = self.input_data_user()
            y_train, y_test = self.sanitize_train_test(y_train, y_test)

            self.input_shape = x_train.shape[1:]
            self.train_set = TensorDataset(torch.Tensor(x_train), self.clean_y(y_train))
            self.test_set = TensorDataset(torch.Tensor(x_test), self.clean_y(y_test))
            print("got base data..")

        elif self.mode == "defense":
            (x_train, y_train, l_train) = self.input_data_defender()
            y_train, l_train = self.sanitize_train_test(y_train, l_train)

            # load the TRAINED base model
            input_shape=x_train.shape[1:]
            self.model = LocationModel.LocationMLP(input_shape, self.output_dims, dropout_probability=0.0)
            model_dict = torch.load(f'saves_new/{self.expt}/Location/base/training_epoch200.pkl')
            self.model.load_state_dict(model_dict)
            self.model.eval()

            # self.criterion=nn.CrossEntropyLoss(reduction='mean')
            # class_loss = self.criterion(output, labels).item()
            # pred = output.data.max(1)[1]
            x_train = torch.from_numpy(x_train).float()
            f_train = self.model(x_train)
            f_train, _ = torch.sort(f_train, axis=1)
            f_train = f_train.detach()

            self.input_shape = f_train.shape[1:]
            self.output_dims = 1

            l_train = torch.tensor(np.expand_dims(l_train, axis=1), dtype=torch.float32)

            print(f_train)
            print(l_train)

            print(f_train.shape)
            print(l_train.shape)

            self.train_set = TensorDataset(f_train, l_train)
            # don't care abt test performance
            self.test_set = TensorDataset(f_train, l_train)
        
        elif self.mode == "defense_eval":
            (x_evaluate,y_evaluate,l_evaluate) = self.input_data_attacker_evaluate()
            y_evaluate, l_evaluate = self.sanitize_train_test(y_evaluate, l_evaluate)

            # access from outside
            self.x_eval = torch.from_numpy(x_evaluate).float()
            self.y_eval = torch.from_numpy(y_evaluate).float()
            self.l_eval = torch.from_numpy(l_evaluate).float()
            
        elif self.mode == 'query' or (self.mode == 'cal' and self.use_own):
            (x_train,y_train),(x_test,y_test) = self.input_data_user()
            y_train, y_test = self.sanitize_train_test(y_train, y_test)

            self.test_set = TensorDataset(torch.Tensor(x_test), self.clean_y(y_test))
            self.input_shape = x_train.shape[1:]

            # 1/5 of total training, size 200
            if self.fold > 0 and self.fold <= int(self.config[self.dataset]["num_folds"]):
                self.x_eval = torch.Tensor(x_train[(self.fold - 1)*200 : self.fold*200])
                self.y_eval = self.clean_y(y_train[(self.fold - 1)*200 : self.fold*200])
                self.l_eval = torch.Tensor(np.ones(200))
                self.train_set = TensorDataset(self.x_eval, self.y_eval)

                # load in the trained model and run it on this subset (for sanity checking)
                    
            elif self.fold != 0: # Unincluded fold in training of size 200
                x_eval, y_train_out = self.input_no_train_data()
                self.x_eval = torch.Tensor(x_eval)
                self.y_eval = self.clean_y(self.sanitize_ydata(y_train_out))
                self.l_eval = torch.Tensor(np.zeros(200))
                self.train_set = TensorDataset(self.x_eval, self.y_eval)
            
            else:  # fold 0 / other tabular dataset not supported
                print("error! not supported")
            
            print("got query data..")

        elif self.mode == 'cal':
            # assume calset is also location!
            (x_train, y_train), (x_test,y_test) = self.input_data_user()
            self.input_shape = x_train.shape[1:]
            y_train, y_test = self.sanitize_train_test(y_train, y_test)

            # 3 to 3.5k for train, 4 to 4.5k for test: 1000 unseen examples total
            (x_train_att, y_train_att), (x_test_att, y_test_att) = self.input_data_attacker_shallow_model_adv1()
            x_data = np.concatenate((x_train_att, x_test_att))
            y_train_att, y_test_att = self.sanitize_train_test(y_train_att, y_test_att)
            y_data = np.concatenate((y_train_att, y_test_att))
            trainset_ori = TensorDataset(torch.Tensor(x_data), self.clean_y(y_data))

            # how much gaussian noise to add??
            sigma = np.sqrt(np.var(x_data))
            noise = np.random.normal(0, sigma, x_data.shape)
            x_data_noisy = x_data + noise
            trainset_noise = TensorDataset(torch.Tensor(x_data_noisy), self.clean_y(y_data))

            # fraction of noisy vs non-noisy determined by k
            trainset_ori = Subset(trainset_ori, list(
                range(0, int((100-self.k)/100*len(trainset_ori)))))
            trainset_noise = Subset(trainset_noise, list(
                range(int((100-self.k)/100*len(trainset_noise)), int(len(trainset_noise)))))

            # 1000 examples total
            self.train_set = ConcatDataset(
                [trainset_ori, trainset_noise])

            # test set separate than training set
            self.test_set = TensorDataset(torch.Tensor(x_test), self.clean_y(y_test))
            print("got cal data..")
    
    def sanitize_ydata(self, y_data):
        y_data = y_data.astype(int)
        # return np.eye(self.output_dims, dtype='uint8')[y_data]
        return y_data
    
    def sanitize_train_test(self, y_train, y_test):
        return self.sanitize_ydata(y_train), self.sanitize_ydata(y_test)
    
    # train from 0k to 1k, test from 1k to 2k
    def input_data_user(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]

        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']
        x_train_user=x_data[index_data[int(self.user_training_data_range["start"]):int(self.user_training_data_range["end"])],:]
        x_test_user=x_data[index_data[int(self.user_testing_data_range["start"]):int(self.user_testing_data_range["end"])],:]
        y_train_user=y_data[index_data[int(self.user_training_data_range["start"]):int(self.user_training_data_range["end"])]]
        y_test_user=y_data[index_data[int(self.user_testing_data_range["start"]):int(self.user_testing_data_range["end"])]]
        y_train_user=y_train_user-1.0
        y_test_user=y_test_user-1.0

        return (x_train_user,y_train_user),(x_test_user,y_test_user)
    
    # evaluating output from 0k to 1k, 1k to 2k as above
    def input_data_defender(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]

        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']
        x_train_user=x_data[index_data[int(self.defense_member_data_index_range["start"]):int(self.defense_member_data_index_range["end"])],:]
        x_nontrain_defender=x_data[index_data[int(self.defense_nonmember_data_index_range["start"]):int(self.defense_nonmember_data_index_range["end"])],:]
        y_train_user=y_data[index_data[int(self.defense_member_data_index_range["start"]):int(self.defense_member_data_index_range["end"])]]
        y_nontrain_defender=y_data[index_data[int(self.defense_nonmember_data_index_range["start"]):int(self.defense_nonmember_data_index_range["end"])]]

        x_train_defender=np.concatenate((x_train_user,x_nontrain_defender),axis=0)
        y_train_defender=np.concatenate((y_train_user,y_nontrain_defender),axis=0)
        y_train_defender=y_train_defender-1.0

        label_train_defender=np.zeros([x_train_defender.shape[0]],dtype=np.int)
        label_train_defender[0:x_train_user.shape[0]]=1
        return (x_train_defender,y_train_defender,label_train_defender)

    def input_data_attacker_evaluate(self):
        print("Attacker evaluate member data range: {}".format(self.attacker_evaluate_member_data_range))
        print("Attacker evaluate nonmember data range: {}".format(self.attacker_evaluate_nonmember_data_range))
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']

        x_evaluate_member_attacker=x_data[index_data[int(self.attacker_evaluate_member_data_range["start"]):int(self.attacker_evaluate_member_data_range["end"])],:]
        x_evaluate_nonmember_attacker=x_data[index_data[int(self.attacker_evaluate_nonmember_data_range["start"]):int(self.attacker_evaluate_nonmember_data_range["end"])],:]
        y_evaluate_member_attacker=y_data[index_data[int(self.attacker_evaluate_member_data_range["start"]):int(self.attacker_evaluate_member_data_range["end"])]]
        y_evaluate_nonmumber_attacker=y_data[index_data[int(self.attacker_evaluate_nonmember_data_range["start"]):int(self.attacker_evaluate_nonmember_data_range["end"])]]
        x_evaluate_attacker=np.concatenate((x_evaluate_member_attacker,x_evaluate_nonmember_attacker),axis=0)
        y_evaluate_attacker=np.concatenate((y_evaluate_member_attacker,y_evaluate_nonmumber_attacker),axis=0)
        y_evaluate_attacker=y_evaluate_attacker-1.0
        label_evaluate_attacker=np.zeros([x_evaluate_attacker.shape[0]],dtype=np.int)
        label_evaluate_attacker[0:x_evaluate_member_attacker.shape[0]]=1

        return (x_evaluate_attacker,y_evaluate_attacker,label_evaluate_attacker)
    
    # 2000 to 3000
    def input_no_train_data(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]

        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']
        x_nonmember_train=x_data[index_data[int(self.attacker_nonmember_even["start"]):int(self.attacker_nonmember_even["end"])],:]
        y_nonmember_train=y_data[index_data[int(self.attacker_nonmember_even["start"]):int(self.attacker_nonmember_even["end"])]]
        y_nonmember_train=y_nonmember_train-1.0

        return x_nonmember_train, y_nonmember_train
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


    def input_data_attacker_shallow_model_adv1(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']

        x_train_member_attacker=x_data[index_data[int(self.attacker_train_member_data_range["start"]):int(self.attacker_train_member_data_range["end"])],:]
        x_train_nonmember_attacker=x_data[index_data[int(self.attacker_train_nonmember_data_range["start"]):int(self.attacker_train_nonmember_data_range["end"])],:]
        y_train_member_attacker=y_data[index_data[int(self.attacker_train_member_data_range["start"]):int(self.attacker_train_member_data_range["end"])]]
        y_train_nonmumber_attacker=y_data[index_data[int(self.attacker_train_nonmember_data_range["start"]):int(self.attacker_train_nonmember_data_range["end"])]]
        y_train_member_attacker=y_train_member_attacker-1.0
        y_train_nonmumber_attacker=y_train_nonmumber_attacker-1.0

        return (x_train_member_attacker,y_train_member_attacker),(x_train_nonmember_attacker,y_train_nonmumber_attacker)