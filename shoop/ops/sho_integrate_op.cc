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
  .Input("num: int64")
  .Output("t: T")
  .Output("x: T")
  .Output("v: T")
  .Output("a: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle s;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &s));
    c->set_output(0, c->UnknownShape());
    c->set_output(1, c->UnknownShape());
    c->set_output(2, c->UnknownShape());
    c->set_output(3, c->UnknownShape());

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
    const Tensor& N_tensor  = context->input(3);

    OP_REQUIRES(context, x0_tensor.dims() == 0, errors::InvalidArgument("x0 must be a scalar"));
    OP_REQUIRES(context, v0_tensor.dims() == 0, errors::InvalidArgument("v0 must be a scalar"));
    OP_REQUIRES(context, k_tensor.dims() == 0, errors::InvalidArgument("k must be a scalar"));
    OP_REQUIRES(context, N_tensor.dims() == 0, errors::InvalidArgument("num must be a scalar"));

    const T x0 = x0_tensor.template scalar<T>()(0);
    const T v0 = v0_tensor.template scalar<T>()(0);
    const T k  = k_tensor.template scalar<T>()(0);
    const int64 N  = N_tensor.scalar<int64>()(0);
    OP_REQUIRES(context, N > 0, errors::InvalidArgument("N must be at least 1"));

    // Build the output grids for the position, velocity, and acceleration
    Tensor* t_tensor = NULL;
    Tensor* x_tensor = NULL;
    Tensor* v_tensor = NULL;
    Tensor* a_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &t_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N}), &x_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N}), &v_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N}), &a_tensor));
    auto t = t_tensor->template flat<T>();
    auto x = x_tensor->template flat<T>();
    auto v = v_tensor->template flat<T>();
    auto a = a_tensor->template flat<T>();

    t(0) = T(0.0);
    x(0) = x0;
    v(0) = v0;
    a(0) = get_acceleration(k, x(0));

    for (int64 n = 1; n < N; ++n) {

      T vhalf = v(n-1) + 0.5 * a(n-1) * step_size_;
      x(n) = x(n-1) + vhalf * step_size_;
      a(n) = get_acceleration(k, x(n));
      v(n) = vhalf + 0.5 * a(n) * step_size_;
      t(n) = t(n-1) + step_size_;

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
