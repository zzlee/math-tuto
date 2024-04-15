#include <stdio.h>

typedef unsigned short uint16_t;

__device__ void cvt_yuyv10c_2_yuyv16(uchar1* src, uint16_t* dst) {
	uchar1 s1 = src[1];
	uchar1 s2 = src[2];
	uchar1 s3 = src[3];

	dst[0] = (uint16_t)(int(s1.x & 0x03) << 8) |
		int((src[0].x & 0xFF));

	dst[1] = (uint16_t)(int(s2.x & 0x0F) << 6) |
		int((s1.x & 0xFC) >> 2);

	dst[2] = (uint16_t)(int(s3.x & 0x3F) << 4) |
		int((s2.x & 0xF0) >> 4);

	dst[3] = (uint16_t)(int(src[4].x & 0xFF) << 2) |
		int((s3.x & 0xC0) >> 6);
}

__device__ void cvt_10_8(uint16_t* src, uint16_t* dst) {
#if 0
	*dst = (uint16_t)((int)*src * 0xFF / 0x3FF);
#else
	*dst = (*src >> 2);
#endif
}

__device__ void endian_swap(uint16_t* src, uint16_t* dst) {
#if 1
		uint16_t a = *dst;

		*src = (int16_t)(((a & 0xFF00) >> 8) | ((a & 0x00FF) << 8));
#endif
}

__global__ void kernel_YUYV_10c_16s_C2R(uchar1* pSrc, int srcStep,
	uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int nSrcIdx = y * srcStep + x * 5;
		int nDstIdx = y * dstStep + x * 8;

		cvt_yuyv10c_2_yuyv16(&pSrc[nSrcIdx + 0], (uint16_t*)&pDst[nDstIdx + 0]);
	}
}

cudaError_t zppiYUYV_10c_16u_C2R(
	const void* pSrc, int srcStep,
	void* pDst, int dstStep, int nWidth, int nHeight) {
	const int BLOCK_W = 16;
	const int BLOCK_H = 16;

	nWidth /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YUYV_10c_16s_C2R<<<grid, block>>>(
		(uchar1*)pSrc, srcStep, (uchar1*)pDst, dstStep, nWidth, nHeight);

	return cudaDeviceSynchronize();
}

__global__ void kernel_Narrow10to8_16u_C2R(uchar1* pSrc, int srcStep,
	uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int nSrcIdx = y * srcStep + x * 2;
		int nDstIdx = y * dstStep + x * 2;

		cvt_10_8((uint16_t*)&pSrc[nSrcIdx + 0], (uint16_t*)&pDst[nDstIdx + 0]);
	}
}

cudaError_t zppiNarrow10to8_16u_C2R(
	const void* pSrc, int srcStep,
	void* pDst, int dstStep, int nWidth, int nHeight) {
	const int BLOCK_W = 16;
	const int BLOCK_H = 16;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_Narrow10to8_16u_C2R<<<grid, block>>>(
		(uchar1*)pSrc, srcStep, (uchar1*)pDst, dstStep, nWidth, nHeight);

	return cudaDeviceSynchronize();
}

__global__ void kernel_EndianSwap_16u_C2R(uchar1* pSrc, int srcStep,
	uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int nSrcIdx = y * srcStep + x * 2;
		int nDstIdx = y * dstStep + x * 2;

		endian_swap((uint16_t*)&pSrc[nSrcIdx + 0], (uint16_t*)&pDst[nDstIdx + 0]);
	}
}

cudaError_t zppiEndianSwap_16u_C2R(
	const void* pSrc, int srcStep,
	void* pDst, int dstStep, int nWidth, int nHeight) {
	const int BLOCK_W = 16;
	const int BLOCK_H = 16;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_EndianSwap_16u_C2R<<<grid, block>>>(
		(uchar1*)pSrc, srcStep, (uchar1*)pDst, dstStep, nWidth, nHeight);

	return cudaDeviceSynchronize();
}