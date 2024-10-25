import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# from tools.deit_features import deit_tiny_patch_features, deit_small_patch_features, deit_base_patch_features
# from tools.cait_features import cait_xxs24_224_features
from vision_transformer import vit_base

# base_architecture_to_features = {'deit_tiny_patch16_224': deit_tiny_patch_features,
#                                  'deit_small_patch16_224': deit_small_patch_features,
#                                  'deit_base_patch16_224': deit_base_patch_features,
#                                  'cait_xxs24_224': cait_xxs24_224_features,}

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, X):
        if self.training:  # Only apply dropout during training
            self.mask = torch.bernoulli(torch.full_like(X, 1 - self.p))
            return X * self.mask
        else:
            return X * (1 - self.p)
        
class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes,
                 reserve_layers=[],
                 reserve_token_nums=[],
                 use_global=False,
                 use_ppc_loss=False,
                 ppc_cov_thresh=2.,
                 ppc_mean_thresh=2,
                 global_coe=0.3,
                 global_proto_per_class=10,
                 init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.reserve_layers = reserve_layers
        self.reserve_token_nums = reserve_token_nums
        self.use_global = use_global
        self.use_ppc_loss = use_ppc_loss
        self.ppc_cov_thresh = ppc_cov_thresh
        self.ppc_mean_thresh = ppc_mean_thresh
        self.global_coe = global_coe
        self.global_proto_per_class = global_proto_per_class
        self.epsilon = 1e-4
        
        self.reserve_layer_nums = list(zip(self.reserve_layers, self.reserve_token_nums))

        self.num_prototypes_global = self.num_classes * self.global_proto_per_class
        self.prototype_shape_global = [self.num_prototypes_global] + self.prototype_shape[1:]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        self.prototype_class_identity_global = torch.zeros(self.num_prototypes_global,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        num_prototypes_per_class_global = self.num_prototypes_global // self.num_classes
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[j, j // num_prototypes_per_class_global] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('MYVISION'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Linear)][-1].out_features
        elif features_name.startswith('MYCAIT'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Linear)][-1].out_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        self.num_patches = self.features.patch_embed.num_patches

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        if self.use_global:
            self.prototype_vectors_global = nn.Parameter(torch.rand(self.prototype_shape_global),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer_global = nn.Linear(self.num_prototypes_global, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer.weight.requires_grad = False
        self.last_layer_global.weight.requires_grad = False

        self.all_attn_mask = None
        self.teacher_model = None

        self.scale = self.prototype_shape[1] ** -0.5

        if init_weights:
            self._initialize_weights()

        # # 遍历 protopformer.features.blocks
        # # 首先关闭 self.protopformer.features 中所有参数的梯度
        # for param in self.features.parameters():
        #     param.requires_grad = False

        # # 然后重新开启最后一个block的梯度
        # for param in self.features.blocks[-1].parameters():
        #     param.requires_grad = True

    def conv_features(self, x, reserve_layer_nums=[]):
        '''
        the feature input to prototype layer
        '''
        batch_size = x.shape[0]
        feature_module_name = self.features.__class__.__name__
        if 'Vision' in feature_module_name or 'MyCait' in feature_module_name:
            if self.use_global:
                cls_embed, x_embed = self.features.forward_feature_patch_embed_all(x)
            else:
                x_embed = self.features.forward_feature_patch_embed(x)
            fea_size, dim = int(x_embed.shape[1] ** (1/2)), x_embed.shape[-1]

            token_attn = None
            x, (cls_token_attn, _) = self.features.forward_feature_mask_train_direct(cls_embed, x_embed, token_attn, reserve_layer_nums)
            final_reserve_num = reserve_layer_nums[-1][1]
            final_reserve_indices = torch.topk(cls_token_attn, k=final_reserve_num, dim=-1)[1]  # (B, final_reserve_num)
            final_reserve_indices = final_reserve_indices.sort(dim=-1)[0]
            final_reserve_indices = final_reserve_indices[:, :, None].repeat(1, 1, dim) # (B, final_reserve_num, dim)

            cls_tokens, img_tokens = x[:, :1], x[:, 1:]   # (B, 1, dim), (B, 196, dim)
            img_tokens = torch.gather(img_tokens, 1, final_reserve_indices)   # (B, final_reserve_num, dim)

            B, dim, fea_len = img_tokens.shape[0], img_tokens.shape[2], img_tokens.shape[1]
            fea_width, fea_height = int(fea_len ** (1/2)), int(fea_len ** (1/2))
            cls_tokens = cls_tokens.permute(0, 2, 1).reshape(B, dim, 1, 1)  # (batch_size, dim, 1, 1)
            img_tokens = img_tokens.permute(0, 2, 1).reshape(B, dim, fea_height, fea_width) # (batch_size, dim, fea_size, fea_size)
        else:
            x = self.features(x)
        
        cls_tokens = self.add_on_layers(cls_tokens)
        img_tokens = self.add_on_layers(img_tokens)
        return (cls_tokens, img_tokens), (token_attn, cls_token_attn, None)

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution_single(self, x, prototype_vectors):
        temp_ones = torch.ones(prototype_vectors.shape).cuda()

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=temp_ones)

        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x, reserve_layer_nums=[]):
        '''
        x is the raw input
        '''
        if self.use_global:
            (cls_tokens, img_tokens), auxi_item = self.conv_features(x, reserve_layer_nums)
            return (cls_tokens, img_tokens), auxi_item

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def get_activations(self, tokens, prototype_vectors):
        batch_size, num_prototypes = tokens.shape[0], prototype_vectors.shape[0]
        distances = self._l2_convolution_single(tokens, prototype_vectors)
        activations = self.distance_2_similarity(distances)   # (B, 2000, 1, 1)
        total_proto_act = activations
        fea_size = activations.shape[-1]
        if fea_size > 1:
            activations = F.max_pool2d(activations, kernel_size=(fea_size, fea_size))   # (B, 2000, 1, 1)
        activations = activations.reshape(batch_size, num_prototypes)
        if self.use_global:
            return activations, (distances, total_proto_act)
        return activations

    def batch_cov(self, points, weights):
        B, N, D = points.size() # weights : (B, N)
        weights = weights / weights.sum(dim=-1, keepdim=True) * N    # (B, N)
        mean = (points * weights[:, :, None]).mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        prods = prods * weights[:, :, None, None]
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return mean, bcov  # (B, D, D)

    def get_PPC_loss(self, total_proto_act, cls_attn_rollout, original_fea_len, label):
        batch_size, original_fea_size = total_proto_act.shape[0], int(original_fea_len ** (1/2))
        proto_per_class = self.num_prototypes_per_class
        discrete_values = torch.FloatTensor([[x, y] for x in range(original_fea_size) for y in range(original_fea_size)]).cuda() # (196, 2)
        discrete_values = discrete_values[None, :, :].repeat(batch_size * proto_per_class, 1, 1)    # (B * 10, 196, 2)
        discrete_weights = torch.zeros(batch_size, proto_per_class, original_fea_len).cuda()   # (B, 10, 196)
        total_proto_act = total_proto_act.flatten(start_dim=2)  # (B, 2000, 81)
        final_token_num = total_proto_act.shape[-1] # 81
        # select the prototypes corresponding to the label
        proto_indices = (label * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        proto_indices += torch.arange(proto_per_class).cuda()   # (B, 10), get 10 indices of activation maps of each sample
        proto_indices = proto_indices[:, :, None].repeat(1, 1, final_token_num)
        total_proto_act = torch.gather(total_proto_act, 1, proto_indices)   # (B, 10, 81)

        reserve_token_indices = torch.topk(cls_attn_rollout, k=final_token_num, dim=-1)[1]   # (B, 81)
        reserve_token_indices = reserve_token_indices.sort(dim=-1)[0]
        reserve_token_indices = reserve_token_indices[:, None, :].repeat(1, proto_per_class, 1) # (B, 10, 81)
        discrete_weights.scatter_(2, reserve_token_indices, total_proto_act)    #   (B, 10, 196)
        discrete_weights = discrete_weights.reshape(batch_size * proto_per_class, -1)   # (B * 10, 196)
        mean_ma, cov_ma = self.batch_cov(discrete_values, discrete_weights)  # (B * 10, 2, 2)
        # cov loss
        ppc_cov_loss = (cov_ma[:, 0, 0] + cov_ma[:, 1, 1]) / 2
        ppc_cov_loss = F.relu(ppc_cov_loss - self.ppc_cov_thresh).mean()
        # mean loss
        mean_ma = mean_ma.reshape(batch_size, proto_per_class, 2)   # (B, 10, 2)
        mean_diff = torch.cdist(mean_ma, mean_ma)   # (B, 10, 10)
        mean_mask = 1. - torch.eye(proto_per_class).cuda()  # (10, 10)
        ppc_mean_loss = F.relu((self.ppc_mean_thresh - mean_diff) * mean_mask).mean()

        return ppc_cov_loss, ppc_mean_loss

    def forward(self, x):
        reserve_layer_nums = self.reserve_layer_nums
        if not self.training:
            if self.use_global:
                (cls_tokens, img_tokens), (token_attn, cls_token_attn, _) = self.prototype_distances(x, reserve_layer_nums)
                global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
                local_activations, (distances, _) = self.get_activations(img_tokens, self.prototype_vectors)

                logits_global = self.last_layer_global(global_activations)
                logits_local = self.last_layer(local_activations)
                logits = self.global_coe * logits_global + (1. - self.global_coe) * logits_local
                return logits, (cls_token_attn, distances, logits_global, logits_local)

        # re-calculate distances
        if self.use_global:
            (cls_tokens, img_tokens), (student_token_attn, cls_attn_rollout, _) = self.prototype_distances(x, reserve_layer_nums)
            cls_attn_rollout = cls_attn_rollout.detach()    # detach
            # get token attention loss
            batch_size, fea_size, original_fea_size = cls_tokens.shape[0], img_tokens.shape[-1], int(cls_attn_rollout.shape[-1] ** (1/2))
            teacher_token_attn = cls_attn_rollout

            global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
            local_activations, (_, total_proto_act) = self.get_activations(img_tokens, self.prototype_vectors)

            logits_global = self.last_layer_global(global_activations)
            logits_local = self.last_layer(local_activations)
            logits = self.global_coe * logits_global + (1. - self.global_coe) * logits_local
        else:
            distances, (student_token_attn, _, _) = self.prototype_distances(x, reserve_layer_nums)
            # global min pooling
            batch_size, fea_size = distances.shape[0], distances.shape[-1]
            prototype_activations = self.distance_2_similarity(distances)   # (B, 2000, 9, 9)
            total_proto_act = prototype_activations # (B, 2000, 9, 9)

            prototype_activations = F.max_pool2d(prototype_activations,
                                        kernel_size=(fea_size,
                                                    fea_size))
            prototype_activations = prototype_activations.view(-1, self.num_prototypes)

            logits = self.last_layer(prototype_activations)

        attn_loss = torch.zeros(1, device=logits.device)
        # attn_loss = F.mse_loss(teacher_token_attn, student_token_attn, reduction='sum')
        original_fea_len = original_fea_size ** 2

        return logits, (student_token_attn, attn_loss, total_proto_act, cls_attn_rollout, original_fea_len)

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        reserve_layer_nums = self.reserve_layer_nums
        (cls_tokens, img_tokens), (token_attn, cls_token_attn, _) = self.prototype_distances(x, reserve_layer_nums)
        global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
        local_activations, (distances, proto_acts) = self.get_activations(img_tokens, self.prototype_vectors)

        return cls_token_attn, proto_acts

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

        if hasattr(self, 'last_layer_global'):
            positive_one_weights_locations = torch.t(self.prototype_class_identity_global)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            self.last_layer_global.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256, use_bn=False, nlayers=3, hidden_dim=2048):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        return x_proj

class HASHHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=12):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim)) ## new
            layers.append(nn.GELU()) ## new
            self.mlp = nn.Sequential(*layers)
            
        self.hash = nn.Linear(bottleneck_dim, code_dim, bias=False)
        self.bn_h = nn.BatchNorm1d(code_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        h = self.hash(x)

        # if x.shape[0] == 128:
        #     feat = self.bn_h(feat)

        # h = custom_tanh(feat, num=3) 
        # h = nn.Tanh()(h)

        return h

def custom_tanh(x, num=1):
    x = x * num
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

class PPNet_Normal(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 num_classes,
                 use_global=False,
                 use_ppc_loss=False,
                 ppc_cov_thresh=2.,
                 ppc_mean_thresh=2,
                 global_coe=0.3,
                 global_proto_per_class=10,
                 init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 mask_theta=0.1):

        super(PPNet_Normal, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes

        self.use_global = use_global
        self.use_ppc_loss = use_ppc_loss
        self.ppc_cov_thresh = ppc_cov_thresh
        self.ppc_mean_thresh = ppc_mean_thresh
        self.global_coe = global_coe
        self.global_proto_per_class = global_proto_per_class
        self.epsilon = 1e-4

        self.num_prototypes_global = self.num_classes * self.global_proto_per_class
        # self.num_prototypes_global = self.num_classes
        self.prototype_shape_global = [self.num_prototypes_global] + [self.prototype_shape[1]]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        self.prototype_class_identity_global = torch.zeros(self.num_prototypes_global,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        num_prototypes_per_class_global = self.num_prototypes_global // self.num_classes
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[j, j // num_prototypes_per_class_global] = 1


        # this has to be named features to allow the precise loading
        self.features = features


        first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Linear)][-1].out_features

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            # self.add_on_layers = nn.Sequential(
            #     nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
            #     nn.Sigmoid()
            #     )
            self.add_on_layers = nn.Sequential(
                nn.Linear(first_add_on_layer_in_channels, self.prototype_shape[1]),
                nn.GELU()
            )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        if self.use_global:
            self.prototype_vectors_global = nn.Parameter(torch.rand(self.prototype_shape_global),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer_global = nn.Linear(self.num_prototypes_global, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer.weight.requires_grad = False
        self.last_layer_global.weight.requires_grad = False

        self.all_attn_mask = None
        self.teacher_model = None

        self.scale = self.prototype_shape[1] ** -0.5

        if init_weights:
            self._initialize_weights()

        # self.my_last = nn.Linear(self.prototype_shape[1], 100, bias=False)
        # nn.init.kaiming_normal_(self.my_last.weight, mode='fan_out', nonlinearity='relu')

        # self.hash_layer1 = nn.Linear(self.prototype_shape[1], 256)
        # self.hash_layer1_norm = nn.BatchNorm1d(256)
        # nn.init.trunc_normal_(self.hash_layer1.weight, std=.02)
        # nn.init.constant_(self.hash_layer1.bias, 0)

        # self.hash_layer2 = nn.Linear(256, 12, bias=False)
        # self.hash_layer2_norm = nn.BatchNorm1d(12)
        # nn.init.trunc_normal_(self.hash_layer2.weight, std=.02)

        # self.hash_head = nn.Sequential(self.hash_layer1, 
        #                                self.hash_layer1_norm,
        #                                self.hash_layer2,
        #                                self.hash_layer2_norm)
        self.hash_head = HASHHead(in_dim=self.prototype_shape[1])
        # self.drop = nn.Dropout(p = 0.1)
        self.drop = Dropout(p=mask_theta)
        # self.mlp = MLP(in_dim=first_add_on_layer_in_channels)

        # self.hash_center_head = HASHHead(in_dim=self.prototype_shape[1])
        # self.hash_center_head = nn.Sequential(nn.Linear(self.prototype_shape[1], 12), nn.Tanh())

        # self.hash_head = nn.Sequential(self.hash_layer, nn.BatchNorm1d(12))

        # self.openset_head = nn.Linear(1000, 200, bias=False)
        # nn.init.kaiming_normal_(self.openset_head.weight, mode='fan_out', nonlinearity='relu')

        # self.my_last_global = nn.Linear(2000, 100, bias=False)

        # trunc_normal_(self.my_last.weight, std=.02)
        # trunc_normal_(self.my_last_global.weight, std=.02)

        # 遍历 protopformer.features.blocks
        # 首先关闭 self.protopformer.features 中所有参数的梯度
        for param in self.features.parameters():
            param.requires_grad = False

        # 然后重新开启最后一个block的梯度
        for param in self.features.blocks[-1].parameters():
            param.requires_grad = True


    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution_single(self, x, prototype_vectors):
        temp_ones = torch.ones(prototype_vectors.shape).cuda()

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=temp_ones)

        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances


    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def get_activations(self, tokens, prototype_vectors):
        # batch_size, num_prototypes = tokens.shape[0], prototype_vectors.shape[0]
        # distances = self._l2_convolution_single(tokens, prototype_vectors)
        # print(distances)
        # print("distances.shape", distances.shape)
        ##my_distance 
        # tokens_expanded = tokens.squeeze()
        tokens_expanded = F.normalize(tokens, dim=-1)
        # prototype_vectors_expanded = prototype_vectors.squeeze()
        prototype_vectors_expanded = F.normalize(prototype_vectors, dim=-1)
        # print("tokens_expanded.shape", tokens_expanded.shape)
        # print("prototype_vectors_expanded.shape", prototype_vectors_expanded.shape)
        tokens_expanded = tokens_expanded.unsqueeze(1).expand(-1, prototype_vectors.size(0), -1)
        prototype_vectors_expanded = prototype_vectors_expanded.unsqueeze(0).expand(tokens.size(0), -1, -1)
        diff = tokens_expanded - prototype_vectors_expanded
        diff_squared = diff ** 2
        euclidean_distances = diff_squared.sum(dim=2).sqrt()
        # print(euclidean_distances)
        # print("euclidean_distances.shape", euclidean_distances.shape)
        ##my_distance 
        activations = self.distance_2_similarity(euclidean_distances)   # (B, 2000, 1, 1)
        total_proto_act = activations
        # fea_size = activations.shape[-1]
        # if fea_size > 1:
        #     activations = F.max_pool2d(activations, kernel_size=(fea_size, fea_size))   # (B, 2000, 1, 1)
        # activations = activations.reshape(batch_size, num_prototypes)
        if self.use_global:
            return activations, (euclidean_distances, total_proto_act)
        return activations



    def forward(self, x):

        if not self.training:
            ##dino
            cls_tokens, _ = self.features.forward_all(x)
            ##deit
            # all_tokens = self.features.forward_features_all(x)
            # cls_tokens, img_tokens = all_tokens[:, :1], all_tokens[:, 1:]
            # print("cls_tokens.shape", cls_tokens.shape)
            # print("img_tokens.shape", img_tokens.shape)
            ##my_rankstat
            # feat = self.rankstat_head(cls_tokens.squeeze(1))
            # feat = nn.ReLU()(feat) 
            # feat = torch.nn.functional.normalize(feat, dim=-1)
            ##my_rankstat

            # B, dim = cls_tokens.shape[0], cls_tokens.shape[2] 
            # fea_height, fea_width = int(img_tokens.shape[1] ** (1/2)), int(img_tokens.shape[1] ** (1/2))
            # cls_tokens = cls_tokens.permute(0, 2, 1).reshape(B, dim, 1, 1) 
            # img_tokens = img_tokens.permute(0, 2, 1).reshape(B, dim, fea_height, fea_width) 
            # print("cls_tokens.shape", cls_tokens.shape)
            # print("img_tokens.shape", img_tokens.shape)

            # hash_feat = self.hash_head(cls_tokens)

            cls_tokens = self.add_on_layers(cls_tokens)
            # img_tokens = self.add_on_layers(img_tokens)

            ##mylast
            # logits_backbone = self.my_last(cls_tokens.squeeze())
            ##mylast
            hash_feat = self.hash_head(cls_tokens)
            # hash_feat = nn.Tanh()(hash_feat)

            global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
            # local_activations, _ = self.get_activations(img_tokens, self.prototype_vectors)

            logits_global = self.last_layer_global(global_activations)
            # feat = self.rankstat_head(global_activations)
            
            # hash_prototypes = nn.Tanh()(hash_prototypes)
            # logits_local = self.last_layer(local_activations)
            # feat = self.rankstat_head(local_activations)
            # logits = self.global_coe * logits_global + (1. - self.global_coe) * logits_local
            # activations = torch.cat([global_activations, local_activations], dim=1)
            # feature = self.my_last(activations)
            # return (logits_global, logits_backbone), (_, _, logits_global, logits_local), feat
            # openset_logits = self.openset_head(global_activations)
            return logits_global, (_, _, _, logits_global), hash_feat

        # re-calculate distances
        ##dino
        cls_tokens, _ = self.features.forward_all(x)
        # feat = self.mlp(cls_tokens)
        ## deit
        # all_tokens = self.features.forward_features_all(x)
        # cls_tokens, img_tokens = all_tokens[:, :1], all_tokens[:, 1:]
        # print("cls_tokens.shape", cls_tokens.shape)
        # print("img_tokens.shape", img_tokens.shape)
        ##my_rankstat
        # feat = self.rankstat_head(cls_tokens.squeeze(1))
        # feat = nn.ReLU()(feat) 
        # feat = torch.nn.functional.normalize(feat, dim=-1)
        ##my_rankstat

        # B, dim = cls_tokens.shape[0], cls_tokens.shape[2] 
        # fea_height, fea_width = int(img_tokens.shape[1] ** (1/2)), int(img_tokens.shape[1] ** (1/2))
        # cls_tokens = cls_tokens.permute(0, 2, 1).reshape(B, dim, 1, 1) 
        # img_tokens = img_tokens.permute(0, 2, 1).reshape(B, dim, fea_height, fea_width) 
        # print("cls_tokens.shape", cls_tokens.shape)
        # print("img_tokens.shape", img_tokens.shape)

        # hash_feat = self.hash_head(cls_tokens)

        cls_tokens = self.add_on_layers(cls_tokens)
        # img_tokens = self.add_on_layers(img_tokens)

        hash_feat = self.hash_head(cls_tokens)
        # hash_feat = nn.Tanh()(hash_feat)

        ##mylast
        # # print("cls_tokens.shape", cls_tokens.shape)
        # logits_backbone = self.my_last(cls_tokens.squeeze())
        ##mylast

        global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
        # local_activations, _ = self.get_activations(img_tokens, self.prototype_vectors)
        ## drop
        # print("drop!!!")
        global_activations = self.drop(global_activations)
        ## drop
        logits_global = self.last_layer_global(global_activations)
        # feat = self.rankstat_head(global_activations)
        # print(cls_tokens.shape)

        # logits_local = self.last_layer(local_activations)
        # feat = self.rankstat_head(local_activations)
        # logits = self.global_coe * logits_global + (1. - self.global_coe) * logits_local

        # print("global_activations.shape", global_activations.shape)
        # print("local_activations.shape", local_activations.shape)
        # activations = torch.cat([global_activations, local_activations], dim=1)
        # feature = self.my_last(activations)

        # return logits_global, logits_backbone, feat
        # openset_logits = self.openset_head(global_activations)
        return logits_global, hash_feat, global_activations, _

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        reserve_layer_nums = self.reserve_layer_nums
        (cls_tokens, img_tokens), (token_attn, cls_token_attn, _) = self.prototype_distances(x, reserve_layer_nums)
        global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
        local_activations, (distances, proto_acts) = self.get_activations(img_tokens, self.prototype_vectors)

        return cls_token_attn, proto_acts


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

        if hasattr(self, 'last_layer_global'):
            positive_one_weights_locations = torch.t(self.prototype_class_identity_global)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            self.last_layer_global.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                trunc_normal_(m.weight, std=.02)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

class BaseNet(nn.Module):

    def __init__(self, base_architecture, pretrained=True, img_size=224, num_classes=200, init_weights=True):
        super(BaseNet, self).__init__()
        
        self.base_architecture = base_architecture
        self.features = base_architecture_to_features[base_architecture](pretrained=pretrained)
        self.img_size = img_size
        self.num_classes = num_classes
        if 'deit' in base_architecture:
            first_add_on_layer_in_channels = \
                    [i for i in self.features.modules() if isinstance(i, nn.Linear)][-1].out_features
        elif 'resnet' in base_architecture or 'vgg' in base_architecture:
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif 'dense' in base_architecture:
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features

        self.last_layer = nn.Linear(first_add_on_layer_in_channels, self.num_classes,
                                    bias=True) # do not use bias
        if init_weights:
            self._initialize_weights()

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        feature_module_name = self.features.__class__.__name__
        conv_features = self.features.forward_feature_maps(x)
        B, dim, fea_len = conv_features.shape[0], conv_features.shape[2], conv_features.shape[1]
        fea_width, fea_height = int(fea_len ** (1/2)), int(fea_len ** (1/2))
        conv_features = conv_features.permute(0, 2, 1).reshape((B, dim, fea_height, fea_width))

        return conv_features

    def forward(self, x):
        if 'deit' in self.base_architecture:
            x = self.features.forward_features(x)
        else:
            x = self.features(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.shape[0], -1)
        out = self.last_layer(x)

        return out, None

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.last_layer.weight, 1)
        if self.last_layer.bias is not None:
            nn.init.constant_(self.last_layer.bias, 0)


def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    reserve_layers=[],
                    reserve_token_nums=[],
                    use_global=False,
                    use_ppc_loss=False,
                    ppc_cov_thresh=1.,
                    ppc_mean_thresh=2.,
                    global_coe=0.5,
                    global_proto_per_class=10,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    if 'deit' in base_architecture or 'pit' in base_architecture or 'tnt' in base_architecture or 'cait' in base_architecture:
        proto_layer_rf_info = [14, 16, 16, 8.0]
    
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 reserve_layers=reserve_layers,
                 reserve_token_nums=reserve_token_nums,
                 use_global=use_global,
                 use_ppc_loss=use_ppc_loss,
                 ppc_cov_thresh=ppc_cov_thresh,
                 ppc_mean_thresh=ppc_mean_thresh,
                 global_coe=global_coe,
                 global_proto_per_class=global_proto_per_class,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

def construct_PPNet_dino(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    reserve_layers=[],
                    reserve_token_nums=[],
                    use_global=False,
                    use_ppc_loss=False,
                    ppc_cov_thresh=1.,
                    ppc_mean_thresh=2.,
                    global_coe=0.5,
                    global_proto_per_class=10,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck',
                    mask_theta=0.1,
                    pretrain_path=''):
    features = vit_base()
    features.load_state_dict(torch.load(pretrain_path))
    # features.load_state_dict(torch.load('/wang_hp/zhy/gcd-task/pretrained/DINO/dino_vitbase16_pretrain.pth'))
    # features.load_state_dict(torch.load('/home/pszzz/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth'))
    # features.load_state_dict(torch.load('/leonardo_work/IscrC_Fed-GCD/GCD_datasets/dino/dino_vitbase16_pretrain.pth'))
    # features.load_state_dict(torch.load('/home/psawl/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth'))
    # features = deit_base_patch_features(pretrained=pretrained)
    
    return PPNet_Normal(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_classes=num_classes,
                 use_global=use_global,
                 use_ppc_loss=use_ppc_loss,
                 ppc_cov_thresh=ppc_cov_thresh,
                 ppc_mean_thresh=ppc_mean_thresh,
                 global_coe=global_coe,
                 global_proto_per_class=global_proto_per_class,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type,
                 mask_theta=mask_theta)