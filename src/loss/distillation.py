"""Feature-distillation loss.

Compares student token features to frozen-teacher token features, with
configurable prefix-token stripping and an optional projection module to
reconcile shape mismatches (e.g., DINOv3-s16 emits 1 CLS + 4 register tokens
that a vanilla patch-mode ViT student does not produce).
"""

import torch
from torch import nn


class DistillationFeatureLoss(nn.Module):
    """Wrap a base loss (e.g. MSELoss) to compare student/teacher feature tokens.

    Handles the common alignment steps between a trainable student and a
    frozen teacher encoder: dropping prefix tokens (CLS, register tokens),
    an optional projection of student features to teacher dim, and
    detaching + dtype-casting the teacher tensor before the base loss.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        drop_teacher_prefix_tokens: int = 0,
        drop_student_prefix_tokens: int = 0,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.drop_teacher_prefix_tokens = drop_teacher_prefix_tokens
        self.drop_student_prefix_tokens = drop_student_prefix_tokens
        self.projection = projection

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss between student and teacher embeddings."""
        if self.drop_teacher_prefix_tokens:
            teacher = teacher[:, self.drop_teacher_prefix_tokens :, :]
        if self.drop_student_prefix_tokens:
            student = student[:, self.drop_student_prefix_tokens :, :]
        teacher = teacher.detach().to(dtype=student.dtype)
        if self.projection is not None:
            student = self.projection(student)
        return self.base_loss(student, teacher)
