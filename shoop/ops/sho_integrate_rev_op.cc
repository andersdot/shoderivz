#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ShoIntegrateRev")
  .Attr("T: {float, double}")
  .Attr("step_size: float")
  .Input("k: T")
  .Input("x: T")
  .Input("bx: T")
  .Input("bv: T")
  .Input("ba: T")
  .Output("bx0: T")
  .Output("bv0: T")
  .Output("bk: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle s;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &s));

    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(3), &s));
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(4), &s));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(0));
    c->set_output(2, c->input(0));

    return Status::OK();
  });


template <typename T>
class ShoIntegrateRevOp : public OpKernel {
 private:
   float step_size_;

 public:
  explicit ShoIntegrateRevOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("step_size", &step_size_));
  }

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& k_tensor  = context->input(0);
    const Tensor& x_tensor  = context->input(1);
    const Tensor& bx_tensor = context->input(2);
    const Tensor& bv_tensor = context->input(3);
    const Tensor& ba_tensor = context->input(4);

    OP_REQUIRES(context, k_tensor.dims() == 0, errors::InvalidArgument("k must be a scalar"));

    const int64 N = x_tensor.NumElements();
    OP_REQUIRES(context, bx_tensor.NumElements() == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, bv_tensor.NumElements() == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, ba_tensor.NumElements() == N, errors::InvalidArgument("dimension mismatch"));

    const T k  = k_tensor.template scalar<T>()(0);

    const auto x = x_tensor.template flat<T>();
    const auto bx = bx_tensor.template flat<T>();
    const auto bv = bv_tensor.template flat<T>();
    const auto ba = ba_tensor.template flat<T>();

    Tensor* bx0_tensor = NULL;
    Tensor* bv0_tensor = NULL;
    Tensor* bk_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, k_tensor.shape(), &bx0_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, k_tensor.shape(), &bv0_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, k_tensor.shape(), &bk_tensor));
    auto bx0 = bx0_tensor->template scalar<T>();
    auto bv0 = bv0_tensor->template scalar<T>();
    auto bk  = bk_tensor->template scalar<T>();

    T bxn = bx(N-1);
    T bvn = bv(N-1);
    T ban = ba(N-1);
    bk(0) = T(0.0);
    for (int64 n = N-1; n >= 1; --n) {
      //  v(n) = vhalf + 0.5 * a(n) * step_size_;
      T bvhalf = bvn;
      ban += 0.5 * step_size_ * bvn;

      //  a(n) = -k*x(n);  //  get_acceleration(k, x(n));
      bk(0) -= x(n) * ban;
      bxn -= k * ban;

      //  x(n) = x(n-1) + vhalf * step_size_;
      bvhalf += bxn * step_size_;
      bxn += bx(n-1);

      //  T vhalf = v(n-1) + 0.5 * a(n-1) * step_size_;
      bvn = bv(n-1) + bvhalf;
      ban = ba(n-1) + 0.5 * step_size_ * bvhalf;
    }

    bk(0) -= x(0) * ban;
    bxn -= k * ban;

    bx0(0) = bxn;
    bv0(0) = bvn;
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ShoIntegrateRev").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      ShoIntegrateRevOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
