#include "ZzLog.h"
#include "ZzUtils.h"
#include "ZzClock.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <complex>

#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/constants/constants.hpp>

ZZ_INIT_LOG("02_hnet");

using namespace __zz_clock__;

namespace __02_hnet__ {
	struct App;

	typedef std::complex<float> complex;

	void to_string(float* a, int count, std::ostream& o);
	void to_string(complex* a, int count, std::ostream& o);
	void to_string_polar(complex* a, int count, std::ostream& o);
	void zero(complex* a, int count);
	void conj(complex* a, complex* b, int count);
	void mult(complex* a, complex* b, complex* c, int count);
	void add(complex* a, complex* b, complex* c, int count);
	void linear_transform(float* a, complex* b, int count);
	void encode(complex* s, complex* r, complex* x, int s_count, int r_count);
	void decode(complex* s, complex* x, complex* r, int s_count, int r_count);

	struct App {
		int argc;
		char **argv;

		ZzUtils::FreeStack oFreeStack;

		int nInputSize;
		int nOutputSize;

		std::vector<float*> oAllInput;
		std::vector<float*> oAllOutput;

		std::vector<complex*> oAllStimulus;
		std::vector<complex*> oAllResponse;

		complex* pXMatrix;

		App(int argc, char **argv);
		~App();

		int Run();
	};
}

namespace __02_hnet__ {
	void to_string(float* a, int count, std::ostream& o) {
		o << "real[" << count << "]";

		if(count > 0) {
			o << "=[" << a[0];
			for(int i = 1;i < count;i++) {
				o << "," << a[i];
			}
			o << ']';
		}
	}

	void to_string(complex* a, int count, std::ostream& o) {
		o << "complex[" << count << "]";

		if(count > 0) {
			o << "=[(" << a[0].real() << ',' << a[0].imag() << ')';
			for(int i = 1;i < count;i++) {
				o << ",(" << a[i].real() << ',' << a[i].imag() << ')';
			}
			o << ']';
		}
	}

	void to_string_polar(complex* a, int count, std::ostream& o) {
		o << "polar[" << count << "]";

		if(count > 0) {
			o << "=[(" << std::abs(a[0]) << ',' << std::arg(a[0]) << ')';
			for(int i = 1;i < count;i++) {
				o << ",(" << std::abs(a[i]) << ',' << std::arg(a[i]) << ')';
			}
			o << ']';
		}
	}

	void zero(complex* a, int count) {
		static const complex z(0, 0);

		for(int i = 0;i < count;i++) {
			a[i] = z;
		}
	}

	void conj(complex* a, complex* b, int count) {
		for(int i = 0;i < count;i++) {
			b[i] = std::conj(a[i]);
		}
	}

	void mult(complex* a, complex* b, complex* c, int count) {
		for(int i = 0;i < count;i++) {
			c[i] = a[i] * b[i];
		}
	}

	void add(complex* a, complex* b, complex* c, int count) {
		for(int i = 0;i < count;i++) {
			c[i] = a[i] + b[i];
		}
	}

	void linear_transform(float* a, complex* b, int count) {
		for(int i = 0;i < count;i++) {
			b[i] = std::polar(1.0f, a[i] * boost::math::constants::pi<float>());
		}
	}

	void encode(complex* s, complex* r, complex* x, int s_count, int r_count) {
		for(int i = 0;i < s_count;i++) {
			const complex& ss = s[i];
			int r_idx = i * r_count;

			for(int j = 0;j < r_count;j++) {
				x[r_idx + j] += std::conj(ss) * r[j];
			}
		}
	}

	void decode(complex* s, complex* x, complex* r, int s_count, int r_count) {
		complex C(s_count);

		for(int j = 0;j < r_count;j++) {
			complex rr(0, 0);

			for(int i = 0;i < s_count;i++) {
				rr += s[i] * x[i * r_count + j];
			}

			r[j] = rr / C;
		}
	}

	App::App(int argc, char **argv) : argc(argc), argv(argv) {
		// LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	App::~App() {
		// LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	int App::Run() {
		int err = 0;

		nInputSize = 20;
		nOutputSize = 8;

		switch(1) { case 1:
			LOGD("generate input");
			{
				oAllInput.resize(100);
				oFreeStack += [&]() {
					oAllInput.clear();
				};

				boost::mt19937 seed((int)_clk());
				boost::uniform_01<float> dist;
				boost::variate_generator<boost::mt19937&, boost::uniform_01<float> > random(seed, dist);

				for(int i = 0;i < oAllInput.size();i++) {
					float* p = new float[nInputSize];
					oFreeStack += [p]() {
						delete p;
					};

					for(int j = 0;j < nInputSize;j++) {
						p[j] = random();
					}

					oAllInput[i] = p;
				}
			}

			LOGD("generate output");
			{
				oAllOutput.resize(oAllInput.size());
				oFreeStack += [&]() {
					oAllOutput.clear();
				};

				boost::mt19937 seed((int)_clk());
				boost::random::uniform_int_distribution<int> dist(0, 1);
				boost::variate_generator<boost::mt19937&, boost::random::uniform_int_distribution<int> > random(seed, dist);

				for(int i = 0;i < oAllOutput.size();i++) {
					float* p = new float[nOutputSize];
					oFreeStack += [p]() {
						delete p;
					};

					for(int j = 0;j < nOutputSize;j++) {
						p[j] = (float)random() / 2.0;
					}

					oAllOutput[i] = p;
				}
			}

			LOGD("generate stimulus");
			{
				oAllStimulus.resize(oAllInput.size());
				oFreeStack += [&]() {
					oAllStimulus.clear();
				};

				for(int i = 0;i < oAllStimulus.size();i++) {
					complex* p = new complex[nInputSize];
					oFreeStack += [p]() {
						delete[] p;
					};

					linear_transform(oAllInput[i], p, nInputSize);

					oAllStimulus[i] = p;
				}
			}

			LOGD("generate response");
			{
				oAllResponse.resize(oAllOutput.size());
				oFreeStack += [&]() {
					oAllResponse.clear();
				};

				for(int i = 0;i < oAllResponse.size();i++) {
					complex* p = new complex[nOutputSize];
					oFreeStack += [p]() {
						delete[] p;
					};

					linear_transform(oAllOutput[i], p, nOutputSize);

					oAllResponse[i] = p;
				}
			}

			LOGD("generate input");
			{
				pXMatrix = new complex[nInputSize * nOutputSize];
				oFreeStack += [&]() {
					delete[] pXMatrix;
				};

				zero(pXMatrix, nInputSize * nOutputSize);
			}

			for(int i = 0;i < 1;i++) {
				encode(oAllStimulus[i], oAllResponse[i], pXMatrix, nInputSize, nOutputSize);
			}

			to_string(oAllInput[0], nInputSize, std::cout);
			std::cout << std::endl;

			to_string(oAllOutput[0], nOutputSize, std::cout);
			std::cout << std::endl;

			to_string_polar(oAllStimulus[0], nInputSize, std::cout);
			std::cout << std::endl;

			to_string_polar(oAllResponse[0], nOutputSize, std::cout);
			std::cout << std::endl;

			to_string_polar(pXMatrix, nOutputSize, std::cout);
			std::cout << std::endl;

			complex* pResponse = new complex[nOutputSize];
			oFreeStack += [pResponse]() {
				delete[] pResponse;
			};
			decode(oAllStimulus[0], pXMatrix, pResponse, nInputSize, nOutputSize);

			to_string_polar(pResponse, nOutputSize, std::cout);
			std::cout << std::endl;

			ZzUtils::TestLoop([&]() -> int {
				return 0;
			});
		}

		oFreeStack.Flush();

		return err;
	}
}

using namespace __02_hnet__;

int main(int argc, char *argv[]) {
	// LOGD("entering...");

	int err;
	{
		App app(argc, argv);
		err = app.Run();

		// LOGD("leaving...");
	}

	return err;
}
