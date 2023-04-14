
class DefaultConfig(object):

    data_root1 = './train'
    data_root2 = './train40'
    # label_root = './data/label_128'
    num_data = 80000
    crop_size = 64
    noise_level = 1040
    noise_level1 = 1040
    batch_size = 40 # batch size  20
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data

    max_epoch = 50
    # lr = 0.0005  # initial learning rate
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5
    # parser.add_argument("--outf", type=str, default="logs", help='path of log files')
    outf='logs'
    load_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level
    save_model_path = './checkpoints/DPDNN_denoise_sigma%d.pth'%noise_level


opt = DefaultConfig()


























