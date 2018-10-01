# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["sho_integrate"]

import tensorflow as tf
from .tf_utils import load_op_library

ops = load_op_library("ops")


def sho_integrate(x0, v0, k, N, step_size, **kwargs):
    return ops.sho_integrate(x0, v0, k, N, step_size, **kwargs)


@tf.RegisterGradient("ShoIntegrate")
def _sho_integrate_rev(op, *grads):
    x0, v0, k, N = op.inputs
    tgrid, xgrid, vgrid, agrid = op.outputs
    btgrid, bxgrid, bvgrid, bagrid = grads
    bx0, bv0, bk = ops.sho_integrate_rev(k, xgrid, bxgrid, bvgrid, bagrid,
                                         op.get_attr("step_size"))
    return (bx0, bv0, bk, None)
