import torch

from restobox.data.image_dataset import ImageFolderDataset
from restobox.export.export_options import ExportOptions
from restobox.models.common.mirnetv2 import MIRNet_v2_DF
from restobox.models.common.nafnet2 import NAFNet, NAFNetColor
from restobox.models.common.swinir import swinir_real_sr_x8
from restobox.tasks.color.color_image_task import ColorImageTask
from restobox.tasks.color.color_image_task_options import ColorImageTaskOptions
from restobox.tasks.color.color_image_task_utilities import create_color_model
from restobox.tasks.sr.sr_image_task_utilities import create_sr_model
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.tasks.sr.sr_image_task import SrImageTask
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, ScaleFactor
from restobox.training.training_options import TrainingOptions
from restobox.training.training_utilities import disable_warnings

if __name__ == "__main__":

    disable_warnings()

    device = torch.device("cuda:1")

    training_options = TrainingOptions(8,100,"./output",compile_model=False,use_amp=True,profile=False,checkpoint_every_n_steps=10)
    optimization_options = OptimizationOptions()
    export_options = ExportOptions(optimize=False)

    color_options = ColorImageTaskOptions(training_options,optimization_options,export_options,resize_size=(256,256),use_lab=False)

    root = MIRNet_v2_DF(inp_channels=1,out_channels=2 if color_options.use_lab else 3)
    #root = NAFNetColor(out_channels=3)
    model = create_color_model(root,color_options.use_lab,color_options.resize_size,device)

    dataset = ImageFolderDataset([
        "/run/media/bglueck/Data/datasets/images_512x512"])

    task = ColorImageTask(dataset,model,color_options,device)

    task.train("color_test01") 