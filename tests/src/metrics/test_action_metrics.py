"""Unit tests for action metrics (ActionADE, ActionFDE, ActionMAE, GripperL1)."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import pytest
import torch

from metrics.action_displacement import ActionADE, ActionFDE
from metrics.action_mae import ActionMAE
from metrics.gripper_accuracy import GripperL1


@pytest.mark.unit
class TestActionADE:
    """Average Displacement Error over trajectory."""

    def test_zero_error(self):
        m = ActionADE(pos_dim=3)
        traj = torch.randn(4, 10, 3)  # (B, T, pos_dim)
        m.update(traj, traj)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_known_displacement(self):
        m = ActionADE(pos_dim=2)
        pred = torch.zeros(2, 5, 2)
        target = torch.zeros(2, 5, 2)
        target[:, :, 0] = 1.0  # all steps off by L2=1
        m.update(pred, target)
        assert m.compute().item() == pytest.approx(1.0, rel=1e-4)

    def test_reset(self):
        m = ActionADE(pos_dim=2)
        pred = torch.zeros(2, 5, 2)
        target = torch.ones(2, 5, 2)
        m.update(pred, target)
        m.reset()
        m.update(pred, pred)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
class TestActionFDE:
    """Final Displacement Error (last time step only)."""

    def test_zero_error(self):
        m = ActionFDE(pos_dim=3)
        traj = torch.randn(4, 10, 3)
        m.update(traj, traj)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_only_last_step_matters(self):
        m = ActionFDE(pos_dim=2)
        pred = torch.zeros(2, 5, 2)
        target = torch.zeros(2, 5, 2)
        target[:, -1, 0] = 3.0  # only last step off by 3
        m.update(pred, target)
        assert m.compute().item() == pytest.approx(3.0, rel=1e-4)


@pytest.mark.unit
class TestActionMAE:
    def test_zero_error(self):
        m = ActionMAE()
        x = torch.randn(4, 7)
        m.update(x, x)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_known_mae(self):
        m = ActionMAE()
        pred = torch.zeros(2, 4)
        target = torch.ones(2, 4)
        m.update(pred, target)
        assert m.compute().item() == pytest.approx(1.0, rel=1e-4)


@pytest.mark.unit
class TestGripperL1:
    def test_zero_error(self):
        m = GripperL1(gripper_idx=-1)
        x = torch.randn(4, 7)
        m.update(x, x)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-5)

    def test_known_l1(self):
        m = GripperL1(gripper_idx=-1)
        pred = torch.zeros(4, 7)
        target = torch.zeros(4, 7)
        target[:, -1] = 2.0
        m.update(pred, target)
        assert m.compute().item() == pytest.approx(2.0, rel=1e-4)
