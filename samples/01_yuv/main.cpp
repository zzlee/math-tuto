#include "ZzLog.h"

#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <cstring>

#ifdef __AVX__
#include <emmintrin.h>
#endif

ZZ_INIT_LOG("01_yuv");

namespace __01_yuv__ {
	enum {
		YCBCR_JPEG,
		YCBCR_601,
		YCBCR_709,
		YCBCR_601_FullRange,
		YCBCR_709_FullRange,
		YCBCR_2020,
		YCBCR_2020_FullRange,

		YCBCR_BUTT,
	};

	enum {
		PRECISION = 6,
		PRECISION_FACTOR = (1<<PRECISION),
	};

	struct rgb2yuv_t;
	struct yuv2rgb_t;

	// |Y|   |y_shift|                        |matrix[0][0] matrix[0][1] matrix[0][2]|   |R|
	// |U| = |  128  | + 1/PRECISION_FACTOR * |matrix[1][0] matrix[1][1] matrix[1][2]| * |G|
	// |V|   |  128  |                        |matrix[2][0] matrix[2][1] matrix[2][2]|   |B|
	struct rgb2yuv_t
	{
		uint8_t y_shift;
		int16_t matrix[3][3];
	};

	// Hyper ref: https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/
	// |R|                        |y_factor      0       v_r_factor|   |Y-y_shift|
	// |G| = 1/PRECISION_FACTOR * |y_factor  u_g_factor  v_g_factor| * |  U-128  |
	// |B|                        |y_factor  u_b_factor      0     |   |  V-128  |
	struct yuv2rgb_t
	{
		short y_shift;
		short y_factor;
		short v_r_factor;
		short u_g_factor;
		short v_g_factor;
		short u_b_factor;
		short rgb_shift;

		//double y_shift_f;
		double y_factor_f;
		double v_r_factor_f;
		double u_g_factor_f;
		double v_g_factor_f;
		double u_b_factor_f;

		yuv2rgb_t();
		yuv2rgb_t(double kR, double kB, bool bSrcFullRange, bool bDstFullRange = true);
		void ccvt(short in[3], short out[3]);
	};

	template<class T> short v(T v);
	template<class T> short clamp(T v, T _min = 0, T _max = 255);

	template<class T> short v(T v) {
		return (short)((v * PRECISION_FACTOR) + 0.5);
	}

	template<class T> short clamp(T v, T _min, T _max) {
		return (short)((v < _min ? _min : v) > _max ? _max : v);
	}

	inline uint8_t clampU8(int64_t v) {
		v = (v+128*PRECISION_FACTOR)>>PRECISION;

		return v > 255 ? 255 : v < 0 ? 0 : v;
	}

	// divide by PRECISION_FACTOR and clamp to [0:255] interval
	// input must be in the [-128*PRECISION_FACTOR:384*PRECISION_FACTOR] range
	inline uint8_t clampU8(int32_t v)
	{
		static const uint8_t lut[512] =
		{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,
		47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
		91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,
		126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,
		159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
		192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,
		225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,
		255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
		255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
		255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
		255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
		};
		return lut[(v+128*PRECISION_FACTOR)>>PRECISION];
	}

	inline uint8_t clampU8(double v) {
		return v > 255 ? 255 : v < 0 ? 0 : v;
	}

	inline uint8_t clampU8_1(int32_t v) {
		v >>= PRECISION;
		return v > 255 ? 255 : v < 0 ? 0 : v;
	}

	inline double clampd(double v) {
		return v < 0 ? 0 : (v > 1 ? 1 : v);
	}

	void invert_mat3x3(const double * src, double * dst);
	void vec3_x_mat3x3(double* a, double* b, double* c);
	void mat3x3_x_mat3x3(double* a, double* b, double* c);
	void mat3x3_x_vec3x1(double* a, double* b, double* c);
	void mat3x3_x_vec3x1(double* a, int* b, double* c);
	void mat3x3_x_vec3x1(int a[3][3], int b[3], int c[3]);
	void ycbcr_limited_range(int a[3], double b[3]);
	void ycbcr_full_range(int a[3], double b[3]);

#if 0
	struct ccvt_t {
		yuv2rgb_t* YUV2RGB[YCBCR_BUTT];

		ccvt_t();
		~ccvt_t();
	} ccvt;
#endif

	struct App {
		int argc;
		char **argv;

		App(int argc, char **argv);
		~App();

		int Run();
	};
}

namespace __01_yuv__ {
	yuv2rgb_t::yuv2rgb_t() {
	}

	yuv2rgb_t::yuv2rgb_t(double kR, double kB, bool bSrcFullRange, bool bDstFullRange) {
		double kG = (1 - kR - kB);
		double kY;
		double kU;
		double kV;
		if(bSrcFullRange) {
			kY = 1.0;
			kU = 0.5;
			kV = 0.5;

			y_shift = 0;
		} else {
			kY = (235 - 16) / 255.0;
			kU = (240 - 16) / 255.0 * 0.5;
			kV = (240 - 16) / 255.0 * 0.5;

			y_shift = 16;
		}

		double kRGB;
		if(bDstFullRange) {
			kRGB = 1.0;

			rgb_shift = 0;
		} else {
			kRGB = (235 - 16) / 255.0;

			rgb_shift = (16 * PRECISION_FACTOR);
		}

		//yuv2rgb = [
		//           [1.0 / kY,		0.0 / kU,				(1-kR) / kV],
		//           [1.0 / kY,		-(1-kB) * kB / kG / kU,	-(1-kR) * kR / kG / kV],
		//           [1.0 / kY,		(1-kB) / kU,			0.0 / kV],
		//           ];

		y_factor_f = (kRGB * 1.0 / kY);
		v_r_factor_f = (kRGB * (1-kR) / kV);
		u_g_factor_f = -(kRGB * (1-kB) * kB / kG / kU);
		v_g_factor_f = -(kRGB * (1-kR) * kR / kG / kV);
		u_b_factor_f = (kRGB * (1-kB) / kU);

		y_factor = v(kRGB * 1.0 / kY);
		v_r_factor = v(kRGB * (1-kR) / kV);
		u_g_factor = -v(kRGB * (1-kB) * kB / kG / kU);
		v_g_factor = -v(kRGB * (1-kR) * kR / kG / kV);
		u_b_factor = v(kRGB * (1-kB) / kU);
	}

	// |R|                        |y_factor      0       v_r_factor|   |Y-y_shift|
	// |G| = 1/PRECISION_FACTOR * |y_factor  u_g_factor  v_g_factor| * |  U-128  |
	// |B|                        |y_factor  u_b_factor      0     |   |  V-128  |
	void yuv2rgb_t::ccvt(short in[3], short out[3]) {
		short y_ = in[0] - y_shift;
		short u_ = in[1] - 128;
		short v_ = in[2] - 128;

		out[0] = clamp((y_factor * y_ + v_r_factor * v_) / PRECISION_FACTOR);
		out[1] = clamp((y_factor * y_ + u_g_factor * u_ + v_g_factor * v_) / PRECISION_FACTOR);
		out[2] = clamp((y_factor * y_ + u_b_factor * u_) / PRECISION_FACTOR);
	}

	void invert_mat3x3(const double * src, double * dst)
	{
		double det;

		/* Compute adjoint: */

		dst[0] = + src[4] * src[8] - src[5] * src[7];
		dst[1] = - src[1] * src[8] + src[2] * src[7];
		dst[2] = + src[1] * src[5] - src[2] * src[4];
		dst[3] = - src[3] * src[8] + src[5] * src[6];
		dst[4] = + src[0] * src[8] - src[2] * src[6];
		dst[5] = - src[0] * src[5] + src[2] * src[3];
		dst[6] = + src[3] * src[7] - src[4] * src[6];
		dst[7] = - src[0] * src[7] + src[1] * src[6];
		dst[8] = + src[0] * src[4] - src[1] * src[3];

		/* Compute determinant: */

		det = src[0] * dst[0] + src[1] * dst[3] + src[2] * dst[6];

		/* Multiply adjoint with reciprocal of determinant: */

		det = 1.0f / det;

		dst[0] *= det;
		dst[1] *= det;
		dst[2] *= det;
		dst[3] *= det;
		dst[4] *= det;
		dst[5] *= det;
		dst[6] *= det;
		dst[7] *= det;
		dst[8] *= det;
	}

	//                             | b0 b1 b2 |
	// | c0 c1 c2 | = | a0 a1 a2 | | b3 b4 b5 |
	//                             | b6 b7 b8 |
	void vec3_x_mat3x3(double* a, double* b, double* c) {
		c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
		c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
		c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
	}

	// | c0 c1 c2 | = | a0 a1 a2 | | b0 b1 b2 |
	// | c3 c4 c5 | = | a3 a4 a5 | | b3 b4 b5 |
	// | c6 c7 c8 | = | a6 a7 a8 | | b6 b7 b8 |
	void mat3x3_x_mat3x3(double* a, double* b, double* c) {
		vec3_x_mat3x3(a + 0, b, c + 0);
		vec3_x_mat3x3(a + 3, b, c + 3);
		vec3_x_mat3x3(a + 6, b, c + 6);
	}

	// | c0 |   | a0 a1 a2 | | b0 |
	// | c1 | = | a3 a4 a5 | | b1 |
	// | c2 |   | a6 a7 a8 | | b2 |
	void mat3x3_x_vec3x1(double* a, double* b, double* c) {
		c[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		c[1] = a[3] * b[0] + a[4] * b[1] + a[5] * b[2];
		c[2] = a[6] * b[0] + a[7] * b[1] + a[8] * b[2];
	}

	void mat3x3_x_vec3x1(double* a, int* b, double* c) {
		c[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		c[1] = a[3] * b[0] + a[4] * b[1] + a[5] * b[2];
		c[2] = a[6] * b[0] + a[7] * b[1] + a[8] * b[2];
	}

	// | c0 |   | a00 a01 a02 | | b0 |
	// | c1 | = | a10 a11 a12 | | b1 |
	// | c2 |   | a20 a21 a22 | | b2 |
	void mat3x3_x_vec3x1(int a[3][3], int b[3], int c[3]) {
		c[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
		c[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
		c[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
	}

	void ycbcr_limited_range(int a[3], double b[3]) {
		b[0] = (a[0] - 16.0) / (235.0 - 16.0);
		b[1] = (a[1] - 16.0) / (240.0 - 16.0);
		b[2] = (a[2] - 16.0) / (240.0 - 16.0);
	}

	void ycbcr_full_range(int a[3], double b[3]) {
		b[0] = a[0] / 255.0;
		b[1] = a[1] / 255.0;
		b[2] = a[2] / 255.0;
	}

#if 0
	ccvt_t::ccvt_t() {
		memset(YUV2RGB, 0, sizeof(YUV2RGB));

		// JPEG
		{
			yuv2rgb_t* param = new yuv2rgb_t();
			param->y_factor_f = 1.0;
			param->v_r_factor_f = 1.402;
			param->u_g_factor_f = -0.3441;
			param->v_g_factor_f = -0.7141;
			param->u_b_factor_f = 1.772;

			param->y_shift = 0;
			param->y_factor = V(param->y_factor_f);
			param->v_r_factor = V(param->v_r_factor_f);
			param->u_g_factor = V(param->u_g_factor_f);
			param->v_g_factor = V(param->v_g_factor_f);
			param->u_b_factor = V(param->u_b_factor_f);

			YUV2RGB[YCBCR_JPEG] = param;
		}

		double kR, kB;

		// BT 601
		kR = 0.299;
		kB = 0.114;
		YUV2RGB[YCBCR_601] = new yuv2rgb_t(kR, kB, false);
		YUV2RGB[YCBCR_601_FullRange] = new yuv2rgb_t(kR, kB, true);

		// BT 709
		kR = 0.2126;
		kB = 0.0722;
		YUV2RGB[YCBCR_709] = new yuv2rgb_t(kR, kB, false);
		YUV2RGB[YCBCR_709_FullRange] = new yuv2rgb_t(kR, kB, true);

		// BT 2020
		kR = 0.2627;
		kB = 0.0593;
		YUV2RGB[YCBCR_2020] = new yuv2rgb_t(kR, kB, false);
		YUV2RGB[YCBCR_2020_FullRange] = new yuv2rgb_t(kR, kB, true);
	}

	ccvt_t::~ccvt_t() {
		for(int i = 0;i < YCBCR_BUTT;i++) {
			delete YUV2RGB[i];
		}
	}
#endif

	App::App(int argc, char **argv) : argc(argc), argv(argv) {
		// LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	App::~App() {
		// LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	int App::Run() {
		int err = 0;

#if 0
		switch(1) { case 1:
			int yuv_0[] = {
				0x3F,
				0x66,
				0xF0
			};
			LOGD("yuv_0: 0x%X(%d) 0x%X(%d) 0x%X(%d)",
				yuv_0[0], yuv_0[0],
				yuv_0[1], yuv_0[1],
				yuv_0[2], yuv_0[2]);
			double yuv_0_d[3];
			ycbcr_limited_range(yuv_0, yuv_0_d);
			LOGD("yuv_0_d: %.4f %.4f %.4f", yuv_0_d[0], yuv_0_d[1], yuv_0_d[2]);

			int16_t yuv_0_s[3] = {
				(int16_t)yuv_0[0],
				(int16_t)yuv_0[1],
				(int16_t)yuv_0[2],
			};
			int16_t rgb_0_s[3];
			ccvt.YUV2RGB[YCBCR_709]->ccvt(yuv_0_s, rgb_0_s);
			LOGD("rgb_0_s: %d %d %d", rgb_0_s[0], rgb_0_s[1], rgb_0_s[2]);

			int yuv_1[] = {
				0x36,
				0x61,
				0xFF
			};
			LOGD("yuv_1: 0x%X(%d) 0x%X(%d) 0x%X(%d)",
				yuv_1[0], yuv_1[0],
				yuv_1[1], yuv_1[1],
				yuv_1[2], yuv_1[2]);
			double yuv_1_d[3];
			ycbcr_full_range(yuv_1, yuv_1_d);
			LOGD("yuv_1_d: %.4f %.4f %.4f", yuv_1_d[0], yuv_1_d[1], yuv_1_d[2]);

			int16_t yuv_1_s[3] = {
				(int16_t)yuv_1[0],
				(int16_t)yuv_1[1],
				(int16_t)yuv_1[2],
			};
			int16_t rgb_1_s[3];
			ccvt.YUV2RGB[YCBCR_709_FullRange]->ccvt(yuv_1_s, rgb_1_s);
			LOGD("rgb_1_s: %d %d %d", rgb_1_s[0], rgb_1_s[1], rgb_1_s[2]);

			err = 0;
		}
#endif

		switch(1) { case 1:
			double kR, kB;

			// BT 709
			kR = 0.2126;
			kB = 0.0722;
			yuv2rgb_t yuv2rgb(kR, kB, true, true);

			double mat_yuv2rgb_bt709[9] = {
				yuv2rgb.y_factor_f,                    0, yuv2rgb.v_r_factor_f,
				yuv2rgb.y_factor_f, yuv2rgb.u_g_factor_f, yuv2rgb.v_g_factor_f,
				yuv2rgb.y_factor_f, yuv2rgb.u_b_factor_f,                    0,
			};
			double mat_rgb2yuv_bt709[9];
			invert_mat3x3(mat_yuv2rgb_bt709, mat_rgb2yuv_bt709);

			{
				int pRGB[3] = { 235, 16, 16 };
				int pYUV[3];

				double rgb[] = {
					(pRGB[0] - 16) * 255.0 / (235 - 16),
					(pRGB[1] - 16) * 255.0 / (235 - 16),
					(pRGB[2] - 16) * 255.0 / (235 - 16)
				};
				double yuv[3];

				mat3x3_x_vec3x1(mat_rgb2yuv_bt709, rgb, yuv);

				pYUV[0] = (int)clampU8(yuv[0]) * (235 - 16) / 255 + 16;
				pYUV[1] = (int)clampU8(yuv[1] + 128) * (240 - 16) / 255 + 16;
				pYUV[2] = (int)clampU8(yuv[2] + 128) * (240 - 16) / 255 + 16;

				LOGD("rgb(%d, %d, %d) ==> yuv(%d, %d, %d)",
					pRGB[0], pRGB[1], pRGB[2],
					pYUV[0], pYUV[1], pYUV[2]);
			}

			{
				int pRGB[3] = { 16, 235, 16 };
				int pYUV[3];

				double rgb[] = {
					(pRGB[0] - 16) * 255.0 / (235 - 16),
					(pRGB[1] - 16) * 255.0 / (235 - 16),
					(pRGB[2] - 16) * 255.0 / (235 - 16)
				};
				double yuv[3];

				mat3x3_x_vec3x1(mat_rgb2yuv_bt709, rgb, yuv);

				pYUV[0] = (int)clampU8(yuv[0]) * (235 - 16) / 255 + 16;
				pYUV[1] = (int)clampU8(yuv[1] + 128) * (240 - 16) / 255 + 16;
				pYUV[2] = (int)clampU8(yuv[2] + 128) * (240 - 16) / 255 + 16;

				LOGD("rgb(%d, %d, %d) ==> yuv(%d, %d, %d)",
					pRGB[0], pRGB[1], pRGB[2],
					pYUV[0], pYUV[1], pYUV[2]);
			}

			{
				int pRGB[3] = { 16, 16, 235 };
				int pYUV[3];

				double rgb[] = {
					(pRGB[0] - 16) * 255.0 / (235 - 16),
					(pRGB[1] - 16) * 255.0 / (235 - 16),
					(pRGB[2] - 16) * 255.0 / (235 - 16)
				};
				double yuv[3];

				mat3x3_x_vec3x1(mat_rgb2yuv_bt709, rgb, yuv);

				pYUV[0] = (int)clampU8(yuv[0]) * (235 - 16) / 255 + 16;
				pYUV[1] = (int)clampU8(yuv[1] + 128) * (240 - 16) / 255 + 16;
				pYUV[2] = (int)clampU8(yuv[2] + 128) * (240 - 16) / 255 + 16;

				LOGD("rgb(%d, %d, %d) ==> yuv(%d, %d, %d)",
					pRGB[0], pRGB[1], pRGB[2],
					pYUV[0], pYUV[1], pYUV[2]);
			}

			{
				int pRGB[3] = { 255, 0, 0 };
				int pYUV[3];

				double rgb[] = {
					double(pRGB[0]),
					double(pRGB[1]),
					double(pRGB[2])
				};
				double yuv[3];

				mat3x3_x_vec3x1(mat_rgb2yuv_bt709, rgb, yuv);

				pYUV[0] = (int)clampU8(yuv[0]);
				pYUV[1] = (int)clampU8(yuv[1] + 128);
				pYUV[2] = (int)clampU8(yuv[2] + 128);

				LOGD("rgb(%d, %d, %d) ==> yuv(%d, %d, %d)",
					pRGB[0], pRGB[1], pRGB[2],
					pYUV[0], pYUV[1], pYUV[2]);
			}

			{
				int pRGB[3] = { 0, 255, 0 };
				int pYUV[3];

				double rgb[] = {
					double(pRGB[0]),
					double(pRGB[1]),
					double(pRGB[2])
				};
				double yuv[3];

				mat3x3_x_vec3x1(mat_rgb2yuv_bt709, rgb, yuv);

				pYUV[0] = (int)clampU8(yuv[0]);
				pYUV[1] = (int)clampU8(yuv[1] + 128);
				pYUV[2] = (int)clampU8(yuv[2] + 128);

				LOGD("rgb(%d, %d, %d) ==> yuv(%d, %d, %d)",
					pRGB[0], pRGB[1], pRGB[2],
					pYUV[0], pYUV[1], pYUV[2]);
			}

			{
				int pRGB[3] = { 0, 0, 255 };
				int pYUV[3];

				double rgb[] = {
					double(pRGB[0]),
					double(pRGB[1]),
					double(pRGB[2])
				};
				double yuv[3];

				mat3x3_x_vec3x1(mat_rgb2yuv_bt709, rgb, yuv);

				pYUV[0] = (int)clampU8(yuv[0]);
				pYUV[1] = (int)clampU8(yuv[1] + 128);
				pYUV[2] = (int)clampU8(yuv[2] + 128);

				LOGD("rgb(%d, %d, %d) ==> yuv(%d, %d, %d)",
					pRGB[0], pRGB[1], pRGB[2],
					pYUV[0], pYUV[1], pYUV[2]);
			}
		}

		return err;
	}
}

using namespace __01_yuv__;

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
