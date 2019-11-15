

class DefaultConfig:
    def __init__(self):
        self.gpu_idx = "1"
        self.batch_size = 1
        self.load_size = 286
        self.fine_size = 256
        self.ngf = 64
        self.ndf = 64
        self.input_nc = 1
        self.output_nc = 1
        self.phase = "test"
        self.checkpoint_dir = "./checkpoint"
        self.dataset_dir = "combine3"
        self.L1_lambda = 10.0
        self.use_resnet = True
        self.use_lsgan = True
        self.max_size = 50
        self.lr = 0.0002
        self.beta1 = 0.5
        self.alpha = 0.2
