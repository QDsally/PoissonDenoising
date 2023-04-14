
class DefaultConfig(object):

    data_root = './colorbsd400'
    # label_root = './data/label_128'
    num_data = 200000
    crop_size = 32
    noise_level = 101
    noise_level1 = 101
    batch_size = 40 # batch size
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data

    max_epoch = 100
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5
    # parser.add_argument("--outf", type=str, default="logs", help='path of log files')
    outf='logs'
    load_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level
    save_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level


opt = DefaultConfig()


























