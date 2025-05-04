import torch

from restobox.data.image_dataset import ImageFolderDataset
from restobox.export.export_options import ExportOptions
from restobox.tasks.sr.sr_utilities import create_sr_model
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.tasks.sr.sr_image_task import SrImageTask
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, ScaleFactor
from restobox.training.training_options import TrainingOptions
from restobox.training.training_utilities import disable_warnings

if __name__ == "__main__":

    disable_warnings()

    device = torch.device("cuda:1")

    training_options = TrainingOptions(64,100,"./output",compile_model=False,use_amp=True,profile=False)
    optimization_options = OptimizationOptions()
    export_options = ExportOptions()

    options = SrImageTaskOptions(training_options,optimization_options,export_options,
                                 scales=[ScaleFactor.simple(8,64,0)])

    initial_scale = options.find_scale(0)

    #root = swinir_real_sr_x8()
    #root = MIRNet_v2_SR(scale=initial_scale.factor)
    #root = SRFusion(scale=initial_scale.factor)
    #root = rcan(pretrained=True,scale=initial_scale.factor)
    #root = Swin2SR(upscale=initial_scale.factor)
    #root = NAFNetSR(initial_scale.factor)
    #root = SRResNet(initial_scale.factor,3,3,base_channels=64,num_blocks=8)
    model = create_sr_model(root,initial_scale,device)

    dataset = ImageFolderDataset([
        "/run/media/bglueck/Data/datasets/images_512x512"])

    task = SrImageTask(dataset,model,options,device)

    task.train("test01")