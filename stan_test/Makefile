STAN_PATH = ../../../stan-dev/math

default: test.cc
	g++ -o test test.cc -std=c++11 -O3 \
		-I$(STAN_PATH) \
		-I$(STAN_PATH)/lib/eigen_3.3.3 \
		-I$(STAN_PATH)/lib/boost_1.66.0 \
		-I$(STAN_PATH)/lib/sundials_3.1.0/include
