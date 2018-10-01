# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_op_library", "ops"]

import os
import sysconfig
import tensorflow as tf


def load_op_library(name):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    dirname = os.path.dirname(os.path.abspath(__file__))
    libfile = os.path.join(dirname, name)
    if suffix is not None:
        libfile += suffix
    else:
        libfile += ".so"
    return tf.load_op_library(libfile)


ops = load_op_library("ops")


@tf.RegisterGradient("ShoIntegrate")
def _sho_integrate_rev(op, *grads):
    x0, v0, k, N = op.inputs
    tgrid, xgrid, vgrid, agrid = op.outputs
    btgrid, bxgrid, bvgrid, bagrid = grads
    bx0, bv0, bk = ops.sho_integrate_rev(k, xgrid, bxgrid, bvgrid, bagrid,
                                         op.get_attr("step_size"))
    return (bx0, bv0, bk, None)
