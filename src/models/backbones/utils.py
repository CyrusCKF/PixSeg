from torchvision.models._utils import IntermediateLayerGetter


def replace_getter_layer(
    getter: IntermediateLayerGetter, last_out: bool, second_last_aux: bool
):
    keys = list(getter.return_layers.keys())
    if last_out:
        getter.return_layers[keys[-1]] = "out"
    if second_last_aux is not None:
        getter.return_layers[keys[-2]] = "aux"
