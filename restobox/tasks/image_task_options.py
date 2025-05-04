from dataclasses import dataclass

from restobox.tasks.task_options import TaskOptions

@dataclass(frozen=True)
class ImageTaskOptions(TaskOptions):
    pass