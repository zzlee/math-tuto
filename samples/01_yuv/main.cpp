#include "ZzLog.h"

#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <cstring>

#include <emmintrin.h>

ZZ_INIT_LOG("01_yuv")

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
		uint8_t y_shift;
		int16_t y_factor;
		int16_t v_r_factor;
		int16_t u_g_factor;
		int16_t v_g_factor;
		int16_t u_b_factor;

		//double y_shift_f;
		double y_factor_f;
		double v_r_factor_f;
		double u_g_factor_f;
		double v_g_factor_f;
		double u_b_factor_f;

		yuv2rgb_t();
		yuv2rgb_t(double kR, double kB, bool bFullRange);
	};

	template<class T> int16_t V(T v);

	void invert_mat3x3(const double * src, double * dst);
	void vec3_x_mat3x3(double* a, double* b, double* c);
	void mat3x3_x_mat3x3(double* a, double* b, double* c);
	void mat3x3_x_vec3x1(double* a, double* b, double* c);
	void mat3x3_x_vec3x1(double* a, int* b, double* c);
	void mat3x3_x_vec3x1(int a[3][3], int b[3], int c[3]);

	struct ccvt_t {
		yuv2rgb_t* YUV2RGB[YCBCR_BUTT];
		rgb2yuv_t* RGB2YUV[YCBCR_BUTT];

		ccvt_t();
		~ccvt_t();
	} ccvt;

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

	yuv2rgb_t::yuv2rgb_t(double kR, double kB, bool bFullRange) {
		double kG = (1 - kR - kB);
		double kY;
		double kU;
		double kV;
		if(bFullRange) {
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

		//yuv2rgb = [
		//           [1.0 / kY,		0.0 / kU,				(1-kR) / kV],
		//           [1.0 / kY,		-(1-kB) * kB / kG / kU,	-(1-kR) * kR / kG / kV],
		//           [1.0 / kY,		(1-kB) / kU,			0.0 / kV],
		//           ];

		y_factor_f = (1.0 / kY);
		v_r_factor_f = ((1-kR) / kV);
		u_g_factor_f = -((1-kB) * kB / kG / kU);
		v_g_factor_f = -((1-kR) * kR / kG / kV);
		u_b_factor_f = ((1-kB) / kU);

		y_factor = V(1.0 / kY);
		v_r_factor = V((1-kR) / kV);
		u_g_factor = -V((1-kB) * kB / kG / kU);
		v_g_factor = -V((1-kR) * kR / kG / kV);
		u_b_factor = V((1-kB) / kU);
	}

	template<class T> int16_t V(T v) {
		return (int16_t)((v * PRECISION_FACTOR) + 0.5);
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

	ccvt_t::ccvt_t() {
		memset(YUV2RGB, 0, sizeof(YUV2RGB));
		memset(RGB2YUV, 0, sizeof(RGB2YUV));

		// JPEG
		{
			yuv2rgb_t* param = new yuv2rgb_t();
			param->y_shift = 0;
			param->y_factor = V(1.0);
			param->v_r_factor = V(1.402);
			param->u_g_factor = -V(0.3441);
			param->v_g_factor = -V(0.7141);
			param->u_b_factor = V(1.772);

			param->y_factor_f = 1.0;
			param->v_r_factor_f = 1.402;
			param->u_g_factor_f = -0.3441;
			param->v_g_factor_f = -0.7141;
			param->u_b_factor_f = 1.772;

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
			delete RGB2YUV[i];
		}
	}

	App::App(int argc, char **argv) : argc(argc), argv(argv) {
		LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	App::~App() {
		LOGD("%s(%d):", __FUNCTION__, __LINE__);
	}

	int App::Run() {
		int err;

		switch(1) { case 1:
			LOGD("HERE");
			err = 0;
		}

		return err;
	}
}

using namespace __01_yuv__;

int main(int argc, char *argv[]) {
	LOGD("entering...");

	int err;
	{
		App app(argc, argv);
		err = app.Run();

		LOGD("leaving...");
	}

	return err;
}
