from utils.hidden_layer import HiddenLayerDense


def create_model_from_arrays(model_list):
    h_prev = None
    h_next = None
    for i in range(int(len(model_list) / 2)):
        h_next = HiddenLayerDense('sigm', model_list[i * 2].numpy(), model_list[i * 2 + 1].numpy())
        print("Layer created")
        if h_prev:
            h_next.link_layer_as_prev(h_prev)
            print("Layer linked")
        h_prev = h_next
    return h_next