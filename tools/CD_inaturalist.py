import os
from copy import deepcopy
import numpy as np
from typing import Any, Tuple
from PIL import Image
from torchvision.datasets import INaturalist

inaturalist_root = '/db/pszzz/iNaturalist'

class INaturalist_SUB(INaturalist):
    def __init__(self, root, version='2017', subclassname='', transform=None, target_transform=None, download=False):

        super(INaturalist_SUB, self).__init__(root=root, version=version, target_type='full', transform=transform, target_transform=target_transform, download=download)

        # Check if the provided subclassname is valid
        valid_super_categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
        if subclassname not in valid_super_categories:
            raise ValueError(f"Invalid subclassname '{subclassname}'. Please provide one of {valid_super_categories}.")

        # Filter the dataset based on the provided subclassname
        self._filter_dataset(subclassname)
        self.uq_idxs = np.array(range(len(self)))

    def _filter_dataset(self, subclassname):
        filtered_index = []
        self.index_map = {}
        self.category_num = 0

        for idx, (category_id, fname) in enumerate(self.index): # 0, '89e0fcba3c7fc4319489544181e5992b.jpg'
            category = self.all_categories[category_id] # 'Actinopterygii/Abudefduf saxatilis'
            super_category = category.split('/')[0] # 'Actinopterygii'
            if super_category == subclassname:
                if category_id not in self.index_map:
                    self.index_map[category_id] = self.category_num
                    self.category_num += 1       
                filtered_index.append((self.index_map[category_id], fname))

        self.index = filtered_index
        self.reverse_index_map = {value: key for key, value in self.index_map.items()}
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[self.reverse_index_map[cat_id]], fname))

        if img.mode != 'RGB':
            img = img.convert('RGB')  # 如果不是 RGB 模式，则转换为 RGB 模式
            
        # 使用 cat_id 作为目标
        target = cat_id

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[index], self.uq_idxs[index]

    def __len__(self) -> int:
        return len(self.index)



def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices
    
def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:
        new_index = [dataset.index[i] for i in idxs]
        dataset.index = new_index
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    # 从 dataset.index 中提取所有的类别索引
    class_indices = [index for index, _ in dataset.index]

    # 筛选出包含在 include_classes 中的类别索引的序号
    cls_idxs = [idx for idx, class_index in enumerate(class_indices) if class_index in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    targets = np.array([x for (x, _) in train_dataset.index])
    train_classes = np.unique(targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_inaturalist_datasets(train_transform, 
                          test_transform,
                          subclassname='',
                          train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, 
                          split_train_val=False, 
                          seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = INaturalist_SUB(root=inaturalist_root, subclassname=subclassname, transform=train_transform)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    whole_test_dataset = INaturalist_SUB(root=inaturalist_root, subclassname=subclassname, transform=test_transform)
    test_dataset = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    return train_dataset_labelled, test_dataset, train_dataset_unlabelled

if __name__ == '__main__':

    print("------------------Amphibia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Amphibia', split_train_val=False,
                         train_classes=range(58), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Animalia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Animalia', split_train_val=False,
                         train_classes=range(39), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Arachnida-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Arachnida', split_train_val=False,
                         train_classes=range(28), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Fungi-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Fungi', split_train_val=False,
                         train_classes=range(61), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Mammalia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Mammalia', split_train_val=False,
                         train_classes=range(93), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Mollusca-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Mollusca', split_train_val=False,
                         train_classes=range(47), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Reptilia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Reptilia', split_train_val=False,
                         train_classes=range(145), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')