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
#include <boost/lexical_cast.hpp>

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
	void subt(complex* a, complex* b, complex* c, int count);
	void linear_transform(float* a, complex* b, int count);
	void encode(complex* s, complex* r, complex* x, int s_count, int r_count);
	void decode(complex* s, complex* x, complex* r, int s_count, int r_count);
	int comb(int N, int K, std::vector<int>& indices);
	int expansion_indices(int n, const std::vector<int>& oOrders, std::vector<int>& oIndices, std::vector<int>& oSizes);
	void expand(complex* a, complex* b, int* indices, int b_count, int order);
	void expand_orders(complex* a, complex* b, int* orders, int* indices, int* sizes, int b_count, int o_count);

	struct App {
		int argc;
		char **argv;

		ZzUtils::FreeStack oFreeStack;

		int nSampleCount;
		int nInputSize;
		int nOutputSize;
		std::vector<int> oExpansionOrders;
		std::vector<int> oExpansionIndices;
		std::vector<int> oExpansionSizes;
		int nExpandedInputSize;
		int nLearningTimes;

		std::vector<float*> oAllInput;
		std::vector<float*> oAllOutput;

		std::vector<complex*> oAllStimulus;
		std::vector<complex*> oAllExpanedStimulus;
		std::vector<complex*> oAllResponse;

		complex* pXMatrix;
		complex* pResponse;

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

	void subt(complex* a, complex* b, complex* c, int count) {
		for(int i = 0;i < count;i++) {
			c[i] = a[i] - b[i];
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
#if 0
		complex C(0);
		for(int i = 0;i < s_count;i++) {
			C += std::abs(s[i]);
		}
#else
		complex C(s_count);
#endif

		for(int j = 0;j < r_count;j++) {
			complex rr(0, 0);

			for(int i = 0;i < s_count;i++) {
				rr += s[i] * x[i * r_count + j];
			}

			r[j] = rr / C;
		}
	}

	int comb(int N, int K, std::vector<int>& indices)
	{
		int nCount = 0;
		std::string bitmask(K, 1); // K leading 1's
		bitmask.resize(N, 0); // N-K trailing 0's

		// print integers and permute bitmask
		do {
			for (int i = 0; i < N; ++i) // [0..N-1] integers
			{
				if (bitmask[i]) indices.push_back(i);
			}

			nCount++;
		} while (std::prev_permutation(bitmask.begin(), bitmask.end()));

		return nCount;
	}

	int expansion_indices(int n, const std::vector<int>& oOrders, std::vector<int>& oIndices, std::vector<int>& oSizes) {
		int nTotalSize = 0;

		for(int o = 0;o < oOrders.size();o++) {
			int nSize = comb(n, oOrders[o], oIndices);
			oSizes.push_back(nSize);
			nTotalSize += nSize;
		}

		return nTotalSize;
	}

	void expand(complex* a, complex* b, int* indices, int b_count, int order) {
		for(int i = 0;i < b_count;i += order) {
			complex r = a[indices[i]];

			for(int j = i + 1;j < i + order;j++) {
				r *= a[indices[j]];
			}

			b[i] = r;
		}
	}

	void expand_orders(complex* a, complex* b, int* orders, int* indices, int* sizes, int b_count, int o_count) {
		int i = 0;
		for(int o = 0;o < o_count;o++) {
			int order = orders[o];
			int size = sizes[o];

			expand(a, &b[i], &indices[i], size, order);

			i += size;
		}
	}

	App::App(int argc, char **argv) : argc(argc), argv(argv) {
		// LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	App::~App() {
		// LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	int App::Run() {
		ZzUtils::TestLoop([&]() -> int {
			int err = 0;

			nSampleCount = 3000;
			nInputSize = 30;
			nOutputSize = 1;

			oExpansionOrders.push_back(2);
			oExpansionOrders.push_back(3);
			oExpansionOrders.push_back(4);
			// oExpansionOrders.push_back(5);
			nExpandedInputSize = expansion_indices(nInputSize, oExpansionOrders, oExpansionIndices, oExpansionSizes);

			LOGD("param: %d=>%d, %d", nInputSize, nOutputSize, nExpandedInputSize);
			nLearningTimes = nSampleCount * 1000;

			switch(1) { case 1:
				LOGD("generate input, %d", nSampleCount);
				{
					oAllInput.resize(nSampleCount);
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

				LOGD("generate output, %d", (int)oAllInput.size());
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

				LOGD("generate stimulus, %d", (int)oAllInput.size());
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

				LOGD("generate expanded stimulus, %d", oAllStimulus.size());
				{
					oAllExpanedStimulus.resize(oAllStimulus.size());
					oFreeStack += [&]() {
						oAllExpanedStimulus.clear();
					};

					for(int i = 0;i < oAllExpanedStimulus.size();i++) {
						complex* p = new complex[nExpandedInputSize];
						oFreeStack += [p]() {
							delete[] p;
						};

						expand_orders(oAllStimulus[i], p, &oExpansionOrders[0], &oExpansionIndices[0], &oExpansionSizes[0], nExpandedInputSize, oExpansionOrders.size());

						oAllExpanedStimulus[i] = p;
					}
				}

				LOGD("generate response, %d", (int)oAllOutput.size());
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

					pResponse = new complex[nOutputSize];
					oFreeStack += [&]() {
						delete[] pResponse;
					};
				}

				LOGD("generate xmatrix, %d", nExpandedInputSize * nOutputSize);
				{
					pXMatrix = new complex[nExpandedInputSize * nOutputSize];
					oFreeStack += [&]() {
						delete[] pXMatrix;
					};

					zero(pXMatrix, nExpandedInputSize * nOutputSize);
				}

				LOGD("enhanced learning, %d", nLearningTimes);
				for(int i = 0;i < nLearningTimes;i++) {
					int t = i % oAllExpanedStimulus.size();

					complex* pStimulus = oAllExpanedStimulus[t];
					complex* pDResponse = oAllResponse[t];

					decode(pStimulus, pXMatrix, pResponse, nExpandedInputSize, nOutputSize);
					subt(pDResponse, pResponse, pResponse, nOutputSize);
					encode(pStimulus, pResponse, pXMatrix, nExpandedInputSize, nOutputSize);
				}

				double fMagDiff = 0;
				double fPhaseDiff = 0;
				for(int i = 0;i < oAllExpanedStimulus.size();i++) {
					complex* pStimulus = oAllExpanedStimulus[i];
					complex* pDResponse = oAllResponse[i];

					decode(pStimulus, pXMatrix, pResponse, nExpandedInputSize, nOutputSize);

					float fDMag = std::abs(*pDResponse);
					float fDPhase = std::arg(*pDResponse);

					float fMag = std::abs(*pResponse);
					float fPhase = std::arg(*pResponse);

					fMagDiff += std::abs(fDMag - fMag);
					fPhaseDiff += std::abs(fDPhase - fPhase);

	#if 0
					LOGD("Test %d:", i);

					to_string_polar(pResponse, nOutputSize, std::cout);
					std::cout << std::endl;

					to_string_polar(pDesiredResponse, nOutputSize, std::cout);
					std::cout << std::endl;
	#endif
				}

				fMagDiff /= oAllStimulus.size();
				fPhaseDiff /= oAllStimulus.size();

				LOGD("result: %.4f, %.4f", fMagDiff, fPhaseDiff);
			}

			oFreeStack.Flush();

			return -1;
		});

		return 0;
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
