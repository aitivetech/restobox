import torch

from restobox.data.image_dataset import ImageFolderDataset
from restobox.export.export_options import ExportOptions
from restobox.models.common.cosae import CosAESR, CosAE
from restobox.models.common.edsr import edsr_baseline
from restobox.models.common.mirnetv2 import MIRNet_v2_DF, MIRNet_v2_SR
from restobox.models.common.nafnet2 import NAFNet, NAFNetSR
from restobox.models.common.rcan import rcan
from restobox.models.common.swin2sr_transformers import create_swin2sr
from restobox.models.common.swinir import swinir_real_sr_x8
from restobox.models.sr.drct import DRCT
from restobox.tasks.color.color_image_task_options import ColorImageTaskOptions
from restobox.tasks.sr.sr_image_task_utilities import create_sr_model
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.tasks.sr.sr_image_task import SrImageTask
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, ScaleFactor
from restobox.training.training_options import TrainingOptions
from restobox.training.training_utilities import disable_warnings

if __name__ == "__main__":

    disable_warnings()

    device = torch.device("cuda:1")

    training_options = TrainingOptions(8,1000,"./output",compile_model=True,use_amp=True,profile=False,evaluate_every_n_steps=200,num_workers=8,prefetch_factor=2)
    optimization_options = OptimizationOptions()
    export_options = ExportOptions()

    input_size = 128
    scale_factor = 4

    sr_options = SrImageTaskOptions(training_options,optimization_options,export_options,
                                 scales=[ScaleFactor.simple(scale_factor,input_size,0)])

    initial_scale = sr_options.find_scale(0)

    root = DRCT(image_size=initial_scale.input_size,upscale=initial_scale.factor)
    #root = CosAESR(CosAE(variant='l'),initial_scale.output_size)
    #root = create_swin2sr(initial_scale.input_size,initial_scale.factor)
    #root = swinir_real_sr_x8()
    #root = MIRNet_v2_SR(scale=initial_scale.factor)
    #root = SRFusion(scale=initial_scale.factor)
    #root = rcan(pretrained=False,scale=initial_scale.factor)
    #root = Swin2SR(upscale=initial_scale.factor)
    #root = NAFNetSR(initial_scale.factor)
    #root = SRResNet(initial_scale.factor,3,3,base_channels=64,num_blocks=8)
    #root = edsr_baseline(initial_scale.factor)
    model = create_sr_model(root,initial_scale,device)

    dataset = ImageFolderDataset([
        "/run/media/bglueck/Data/datasets/images_512x512"])

    task = SrImageTask(dataset,model,sr_options,device)

    task.train("test01")