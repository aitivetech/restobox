from argparse import ArgumentParser

import torch

from restobox.data.image_dataset import ImageFolderDataset
from restobox.export.export_options import ExportOptions
from restobox.models.common.cosae import CosAE
from restobox.models.common.mirnetv2 import MIRNet_v2_SR
from restobox.models.common.nafnet import NAFNetSR
from restobox.models.common.srresnet import SRResNet
from restobox.models.common.swinirv2 import Swin2SR
from restobox.models.model import Model
from restobox.models.sr.sr_utilities import create_sr_model
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.tasks.sr.sr_image_task import SrImageTask
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, ScaleFactor
from restobox.training.training_options import TrainingOptions
from restobox.training.training_utilities import disable_warnings

if __name__ == "__main__":

    disable_warnings()

    device = torch.device("cuda:1")

    training_options = TrainingOptions(16,100,"./output",compile_model=True,use_amp=True)
    optimization_options = OptimizationOptions()
    export_options = ExportOptions()

    options = SrImageTaskOptions(training_options,optimization_options,export_options,
                                 scales=[ScaleFactor.simple(4,128,0)])

    initial_scale = options.find_scale(0)

    #root = MIRNet_v2_SR(scale=initial_scale.factor)

    #root = Swin2SR(upscale=initial_scale.factor)
    root = NAFNetSR(initial_scale.factor)
    #root = SRResNet(initial_scale.factor,3,3,base_channels=128,num_blocks=9)
    model = create_sr_model(root,initial_scale,device)

    dataset = ImageFolderDataset([
        "/run/media/bglueck/Data/datasets/open-images-v7/images",
        "/run/media/bglueck/Data/datasets/laion400m/images2"])

    task = SrImageTask(dataset,model,options,device)

    task.train("test01")