#include "ZzLog.h"
#include "ZzUtils.h"
#include "ZzClock.h"

#include <stdint.h>
#include <fstream>
#include <boost/filesystem.hpp>

ZZ_INIT_LOG("03_p210");

using namespace boost::filesystem;
using namespace __zz_clock__;

namespace __03_p210__ {
	struct App;

	union uint32_m {
		uint32_t u32;

		struct {
			uint8_t x, y, z, w;
		} __attribute__ ((aligned (2))) u8;

		struct {
			uint16_t x, y;
		} __attribute__ ((aligned (2))) u16;

		uint8_t u8s[4];
		uint8_t u16s[2];
	};

	union uint16_m {
		uint16_t u16;

		struct {
			uint8_t x, y;
		} __attribute__ ((aligned (2))) u8;

		uint8_t u8s[2];
	};

	union uint40_m {
		struct {
			uint8_t a, b, c, d, e;
		} __attribute__ ((aligned (2))) u8;

		uint8_t u8s[5];
	};

	struct App {
		int argc;
		char **argv;
		int err;
		ZzUtils::FreeStack oFreeStack;

		int nWidth;
		int nHeight;

		std::ifstream ifSrc;
		std::ofstream ofDst;
		std::ofstream ofLog;

		int nSrcStep;
		std::vector<uint8_t> vSrc;
		int nDstStep;
		std::vector<uint8_t> vDst;

		App(int argc, char **argv);
		~App();

		int Run();
	};

	inline uint16_t tohs(uint16_t a) {
#if 1
		return (uint16_t)(((a & 0xFF00) >> 8) | ((a & 0x00FF) << 8));
#else
		return a;
#endif
	}

	inline void cvt_yuyv10c_2_yuyv16(uint40_m* src, uint16_t* dst) {
		dst[0] = (uint16_t)(uint32_t(src->u8s[1] & 0x03) << 8) |
			uint32_t((src->u8s[0] & 0xFF));

		dst[1] = (uint16_t)(uint32_t(src->u8s[2] & 0x0F) << 6) |
			uint32_t((src->u8s[1] & 0xFC) >> 2);

		dst[2] = (uint16_t)(uint32_t(src->u8s[3] & 0x3F) << 4) |
			uint32_t((src->u8s[2] & 0xF0) >> 4);

		dst[3] = (uint16_t)(uint32_t(src->u8s[4] & 0xFF) << 2) |
			uint32_t((src->u8s[3] & 0xC0) >> 6);
	}

	inline uint32_t tohl(uint32_t a) {
		return (uint32_t)(((a & 0xFF000000) >> 24) | ((a & 0x00FF0000) >> 8) | ((a & 0x0000FF00) << 8) | ((a & 0x000000FF) << 24));
	}

	inline uint16_t cvt_8_10(uint8_t a) {
		return (uint16_t)((uint32_t)a * 0x3FF / 0xFF);
	}

	inline uint16_t cvt_8_16(uint8_t a) {
		return (uint16_t)((uint32_t)a * 0xFFFF / 0xFF);
	}

	inline uint8_t cvt_10_8(uint16_t a) {
		return (uint8_t)((uint32_t)a * 0xFF / 0x3FF);
	}
}

namespace __03_p210__ {
	App::App(int argc, char **argv) : argc(argc), argv(argv) {
	}

	App::~App() {
	}

	int App::Run() {
		err = 0;
		nWidth = 1920;
		nHeight = 1080;

		ZzUtils::TestLoop([&]() -> int {
			switch(1) { case 1:
				path oRoot("tests");
				path oSrcPath = oRoot / path("1080.y210-compact");
				path oDstPath = oRoot / path("1080.y210");
				path oLogPath = oRoot / path("1080.y210.log");

				LOGD("oSrcPath=%s", oSrcPath.c_str());
				LOGD("oDstPath=%s", oDstPath.c_str());
				LOGD("oLogPath=%s", oLogPath.c_str());

				ifSrc.open(oSrcPath.c_str(), std::ios::binary);
				if(! ifSrc) {
					LOGE("%s(%d): ifSrc.open() failed", __FUNCTION__, __LINE__);
					break;
				}
				oFreeStack += [&]() {
					ifSrc.close();
				};

				ofDst.open(oDstPath.c_str(), std::ios::binary);
				if(! ofDst) {
					LOGE("%s(%d): ofDst.open() failed", __FUNCTION__, __LINE__);
					break;
				}
				oFreeStack += [&]() {
					ofDst.close();
				};

				ofLog.open(oLogPath.c_str());
				if(! ofLog) {
					LOGE("%s(%d): ofLog.open() failed", __FUNCTION__, __LINE__);
					break;
				}
				oFreeStack += [&]() {
					ofLog.close();
				};

				nSrcStep = nWidth * 2 * 10 / 8;
				vSrc.resize(nSrcStep * nHeight);
				oFreeStack += [&]() {
					vSrc.clear();
				};

				nDstStep = nWidth * 2 * 16 / 8;
				vDst.resize(nDstStep * nHeight);
				oFreeStack += [&]() {
					vDst.clear();
				};

				if(! ifSrc.read((char*)&vSrc[0], vSrc.size())) {
					LOGE("%s(%d): ifSrc.read() failed", __FUNCTION__, __LINE__);
					break;
				}

				LOGD("%d, %d", sizeof(uint32_m), sizeof(uint16_m));

				uint32_m a;
				a.u32 = 0x1234ABCD;
				LOGD("%08X, %08X", a.u32, tohl(a.u32));

				uint32_m b;
				b.u32 = 0x1234;
				LOGD("%04X, %04X", b.u32, tohs(b.u32));

				int nTries = 500;
				LOGD("Starts, nTries=%d", nTries);

				int64_t nBeginTime = _clk();
				for(int i = 0;i < nTries;i++) {
					for(int y = 0;y < nHeight;y++) {
						for(int x = 0;x < nWidth / 2;x++) {
							int nSrcIdx = y * nSrcStep + x * 5;
							int nDstIdx = y * nDstStep + x * 8;

							uint40_m a = *(uint40_m*)&vSrc[nSrcIdx + 0];

							cvt_yuyv10c_2_yuyv16(&a, (uint16_t*)&vDst[nDstIdx + 0]);
						}
					}

#if 1
					// LOGD("cvt_10_8...");
					for(int y = 0;y < nHeight;y++) {
						for(int x = 0;x < nWidth / 2;x++) {
							int nDstIdx = y * nDstStep + x * 8;

							*(uint16_t*)&vDst[nDstIdx + 0] = (uint16_t)cvt_10_8(*(uint16_t*)&vDst[nDstIdx + 0]);
							*(uint16_t*)&vDst[nDstIdx + 2] = (uint16_t)cvt_10_8(*(uint16_t*)&vDst[nDstIdx + 2]);
							*(uint16_t*)&vDst[nDstIdx + 4] = (uint16_t)cvt_10_8(*(uint16_t*)&vDst[nDstIdx + 4]);
							*(uint16_t*)&vDst[nDstIdx + 6] = (uint16_t)cvt_10_8(*(uint16_t*)&vDst[nDstIdx + 6]);
						}
					}
#endif

#if 1
					// LOGD("tohs...");
					for(int y = 0;y < nHeight;y++) {
						for(int x = 0;x < nWidth / 2;x++) {
							int nDstIdx = y * nDstStep + x * 8;

							*(uint16_t*)&vDst[nDstIdx + 0] = tohs(*(uint16_t*)&vDst[nDstIdx + 0]);
							*(uint16_t*)&vDst[nDstIdx + 2] = tohs(*(uint16_t*)&vDst[nDstIdx + 2]);
							*(uint16_t*)&vDst[nDstIdx + 4] = tohs(*(uint16_t*)&vDst[nDstIdx + 4]);
							*(uint16_t*)&vDst[nDstIdx + 6] = tohs(*(uint16_t*)&vDst[nDstIdx + 6]);
						}
					}
#endif
				}
				int64_t nEndTime = _clk();
				LOGD("FPS: %.2f", (nTries * 1000000.0) / (nEndTime - nBeginTime));

				if(! ofDst.write((char*)&vDst[0], vDst.size())) {
					LOGE("%s(%d): ofDst.read() failed", __FUNCTION__, __LINE__);
					break;
				}
			}

			return 1;
		});

		oFreeStack.Flush();

		return err;
	}
}

using namespace __03_p210__;

int main(int argc, char *argv[]) {
	int err;
	{
		App app(argc, argv);
		err = app.Run();
	}

	return err;
}
