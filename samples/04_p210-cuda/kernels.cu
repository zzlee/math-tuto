#include <stdio.h>

enum {
	PRECISION = 6,
	PRECISION_FACTOR = (1<<PRECISION),
};

struct zppi_yuv2rgb_t {
	short y_shift;
	short y_factor;
	short v_r_factor;
	short u_g_factor;
	short v_g_factor;
	short u_b_factor;
};

__device__ short kernel_clamp(short x) {
	return max(0, min(255, x));
}

// |R|                        |y_factor      0       v_r_factor|   |Y-y_shift|
// |G| = 1/PRECISION_FACTOR * |y_factor  u_g_factor  v_g_factor| * |  U-128  |
// |B|                        |y_factor  u_b_factor      0     |   |  V-128  |
__device__ void kernel_yuv_rgb(short y, short u, short v, uchar1 rgb[3], zppi_yuv2rgb_t* param) {
	short y_ = param->y_factor * y;

	rgb[0].x = kernel_clamp((y_ + param->v_r_factor * v) >> PRECISION);
	rgb[1].x = kernel_clamp((y_ + param->u_g_factor * u + param->v_g_factor * v) >> PRECISION);
	rgb[2].x = kernel_clamp((y_ + param->u_b_factor * u) >> PRECISION);
}

__global__ void kernel_YCbCr422_RGB_8u_C2C3R(uchar1* pSrc, int srcStep,
	uchar1* pDst, int dstStep, int nWidth, int nHeight, zppi_yuv2rgb_t* param) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int nSrcIdx = y * srcStep + x * 4;
		int nDstIdx = y * dstStep + x * 6;

		short y0 = (short)pSrc[nSrcIdx + 0].x - (short)param->y_shift;
		short u0 = (short)pSrc[nSrcIdx + 1].x - 128;
		short y1 = (short)pSrc[nSrcIdx + 2].x - (short)param->y_shift;
		short v0 = (short)pSrc[nSrcIdx + 3].x - 128;

		kernel_yuv_rgb(y0, u0, v0, &pDst[nDstIdx + 0], param);
		kernel_yuv_rgb(y1, u0, v0, &pDst[nDstIdx + 3], param);
	}
}

cudaError_t zppiYCbCr422_RGB_8u_C2C3R(
	uchar1* pSrc, int srcStep, uchar1* pDst, int dstStep, int nWidth, int nHeight, zppi_yuv2rgb_t* param) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr422_RGB_8u_C2C3R<<<grid, block>>>(
		pSrc, srcStep,
		pDst, dstStep,
		nWidth / 2, nHeight, param);

	return cudaDeviceSynchronize();
}

