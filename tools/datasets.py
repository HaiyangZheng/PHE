import pickle
import torch

from torchvision import transforms
from tools.CD_cub import get_cub_datasets
from tools.CD_cars import get_scars_datasets
from tools.CD_cifar100 import get_cifar_10_datasets, get_cifar_100_datasets
from tools.CD_herb import get_herbarium_datasets
from tools.CD_food101 import get_food_101_datasets
from tools.CD_aircraft import get_aircraft_datasets
from tools.CD_flowers import get_oxford_flowers_datasets
from tools.CD_pets import get_oxford_pets_datasets
from tools.CD_inaturalist import get_inaturalist_datasets

from copy import deepcopy


def build_dataset(is_train, args):       

    if args.data_set == 'CD_CUB2011U':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        split_path = 'ssb_splits/cub_osr_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        train_dataset, test_dataset, train_dataset_unlabelled = get_cub_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=train_classes, prop_train_labels=args.prop_train_labels, data_root=args.data_root)
        print("train_classes:", train_classes)
        print("len(train_classes):", len(train_classes))
        print("unlabeled_classes:", unlabeled_classes)
        print("len(unlabeled_classes):", len(unlabeled_classes))
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform
        train_dataset_unlabelled.target_transform = target_transform

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_aircraft':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        split_path = 'ssb_splits/aircraft_osr_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        train_dataset, test_dataset, train_dataset_unlabelled = get_aircraft_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=train_classes, prop_train_labels=0.5)
        print("train_classes:", train_classes)
        print("len(train_classes):", len(train_classes))
        print("unlabeled_classes:", unlabeled_classes)
        print("len(unlabeled_classes):", len(unlabeled_classes))
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform
        train_dataset_unlabelled.target_transform = target_transform

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_food':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_food_101_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(51), prop_train_labels=args.prop_train_labels, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_flower':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_oxford_flowers_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(51), prop_train_labels=0.5)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_pets':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_oxford_pets_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(19), prop_train_labels=args.prop_train_labels, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Amphibia':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Amphibia',
                                   train_classes=range(58), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Animalia':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Animalia',
                                   train_classes=range(39), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Arachnida':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Arachnida',
                                   train_classes=range(28), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Fungi':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])


        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Fungi',
                                   train_classes=range(61), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Mammalia':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Mammalia',
                                   train_classes=range(93), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Mollusca':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Mollusca',
                                   train_classes=range(47), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'Reptilia':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Reptilia',
                                   train_classes=range(145), prop_train_labels=0.5, data_root=args.data_root)

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_herb':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.14, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        split_path = 'ssb_splits/herbarium_19_class_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        train_classes = class_info['Old']
        unlabeled_classes = class_info['New']

        train_dataset, test_dataset, train_dataset_unlabelled = get_herbarium_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=train_classes, prop_train_labels=0.5)
        print("train_classes:", train_classes)
        print("len(train_classes):", len(train_classes))
        print("unlabeled_classes:", unlabeled_classes)
        print("len(unlabeled_classes):", len(unlabeled_classes))
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform
        train_dataset_unlabelled.target_transform = target_transform

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_CIFAR100':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_cifar_100_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(50), prop_train_labels=0.5)
        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes

    elif args.data_set == 'CD_CIFAR10':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        train_dataset, test_dataset, train_dataset_unlabelled = get_cifar_10_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=range(5), prop_train_labels=0.5)
        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes
    ####my
    ####my
    elif args.data_set == 'CD_Car':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

        split_path = 'ssb_splits/scars_osr_splits.pkl'
        with open(split_path, 'rb') as handle:
            class_info = pickle.load(handle)

        train_classes = class_info['known_classes']
        open_set_classes = class_info['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        train_dataset, test_dataset, train_dataset_unlabelled = get_scars_datasets(train_transform=transform, test_transform=test_transform, 
                                   train_classes=train_classes, prop_train_labels=args.prop_train_labels, data_root=args.data_root)
        print("train_classes:", train_classes)
        print("len(train_classes):", len(train_classes))
        print("unlabeled_classes:", unlabeled_classes)
        print("len(unlabeled_classes):", len(unlabeled_classes))
        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_dataset.target_transform = target_transform
        test_dataset.target_transform = target_transform
        train_dataset_unlabelled.target_transform = target_transform

        unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
        unlabelled_train_examples_test.transform = test_transform
        nb_classes = args.labeled_nums
        return train_dataset, test_dataset, unlabelled_train_examples_test, nb_classes