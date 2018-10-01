#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ShoIntegrate")
  .Attr("T: {float, double}")
  .Attr("step_size: float")
  .Input("x0: T")
  .Input("v0: T")
  .Input("k: T")
  .Input("tobs: T")
  .Output("xobs: T")
  .Output("vobs: T")
  .Output("tgrid: T")
  .Output("xgrid: T")
  .Output("vgrid: T")
  .Output("agrid: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle x0, v0, k, tobs;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &x0));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &v0));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &k));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &tobs));

    c->set_output(0, tobs);
    c->set_output(1, tobs);
    c->set_output(2, c->UnknownShape());
    c->set_output(3, c->UnknownShape());
    c->set_output(4, c->UnknownShape());
    c->set_output(5, c->UnknownShape());

    return Status::OK();
  });

template <typename T>
inline T get_acceleration (const T& k, const T& x)
{
  return -k * x;
}

template <typename T>
class ShoIntegrateOp : public OpKernel {
 private:
   float step_size_;

 public:
  explicit ShoIntegrateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("step_size", &step_size_));
  }

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& x0_tensor = context->input(0);
    const Tensor& v0_tensor = context->input(1);
    const Tensor& k_tensor  = context->input(2);
    const Tensor& tobs_tensor = context->input(3);

    OP_REQUIRES(context, x0_tensor.dims() == 0, errors::InvalidArgument("x0 must be a scalar"));
    OP_REQUIRES(context, v0_tensor.dims() == 0, errors::InvalidArgument("v0 must be a scalar"));
    OP_REQUIRES(context, k_tensor.dims() == 0, errors::InvalidArgument("k must be a scalar"));

    const int64 N = tobs_tensor.NumElements();

    // Output
    Tensor* xobs_tensor = NULL;
    Tensor* vobs_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, tobs_tensor.shape(), &xobs_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, tobs_tensor.shape(), &vobs_tensor));

    const T x0 = x0_tensor.template scalar<T>()(0);
    const T v0 = v0_tensor.template scalar<T>()(0);
    const T k  = k_tensor.template scalar<T>()(0);
    const auto tobs = tobs_tensor.template flat<T>();
    auto xobs = xobs_tensor->template flat<T>();
    auto vobs = vobs_tensor->template flat<T>();

    // Check that the tobs values are sorted
    OP_REQUIRES(context, tobs(0) >= T(0.0), errors::InvalidArgument("tobs must be sorted and positive"));
    for (int64 n = 1; n < N; ++n)
      OP_REQUIRES(context, tobs(n) >= tobs(n-1), errors::InvalidArgument("tobs must be sorted and positive"));

    // Build the output grids for the position, velocity, and acceleration
    int64 Ngrid = floor(tobs(N-1) / step_size_) + 1;
    Tensor* tgrid_tensor = NULL;
    Tensor* xgrid_tensor = NULL;
    Tensor* vgrid_tensor = NULL;
    Tensor* agrid_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({Ngrid}), &tgrid_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({Ngrid}), &xgrid_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({Ngrid}), &vgrid_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape({Ngrid}), &agrid_tensor));
    auto tgrid = tgrid_tensor->template flat<T>();
    auto xgrid = xgrid_tensor->template flat<T>();
    auto vgrid = vgrid_tensor->template flat<T>();
    auto agrid = agrid_tensor->template flat<T>();

    tgrid(0) = T(0.0);
    xgrid(0) = x0;
    vgrid(0) = v0;
    agrid(0) = get_acceleration(k, xgrid(0));

    int64 j = 0;
    for (int64 n = 0; n < N; ++n) {

      while (tgrid(j) + step_size_ < tobs(n)) {
        ++j;
        vgrid(j-1) += 0.5 * agrid(j-1) * step_size_;
        xgrid(j) = xgrid(j-1) + vgrid(j-1) * step_size_;
        agrid(j) = get_acceleration(k, xgrid(j));
        vgrid(j) = vgrid(j-1) + 0.5 * agrid(j) * step_size_;
        tgrid(j) = tgrid(j-1) + step_size_;
      }

      // A "tiny" step to synchronize to the observed time
      T dt_tiny = tobs(n) - tgrid(j);
      vobs(n) = vgrid(j) + 0.5 * agrid(j) * dt_tiny;
      xobs(n) = xgrid(j) + vobs(n) * dt_tiny;
      vobs(n) += 0.5 * get_acceleration(k, xobs(n)) * dt_tiny;

    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ShoIntegrate").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      ShoIntegrateOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
