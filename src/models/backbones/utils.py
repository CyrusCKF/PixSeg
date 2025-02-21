from torchvision.models._utils import IntermediateLayerGetter


def replace_layer_name(getter: IntermediateLayerGetter, index_to_name: dict[int, str]):
    keys = list(getter.return_layers.keys())
    for i, n in index_to_name.items():
        getter.return_layers[keys[i]] = n
