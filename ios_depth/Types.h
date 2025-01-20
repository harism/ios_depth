#ifndef Types_h
#define Types_h

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#endif

#include <simd/simd.h>

#define VAR_count 262144

struct Vars
{
    float simWidth;
    float d;
    float h;
    float m;
    float p0;
    float gasConstant;
    float viscosityConstant;
    float poly6;
    float poly6Lap;
    float poly6Grad;
    float spikyLap;
    float viscGrad;
    int maxKey;
};

struct KeyIndex
{
    int32_t key;
    int32_t index;
};

struct Particle
{
    int oidx;
    simd_float3 pos;
    simd_float3 vel;
    simd_float3 force;
    float density;
    float pressure;
    simd_float4 color;
};

#ifdef __METAL_VERSION__
int getKey(int3 key);
int3 getKey(float3 pos, float simWidth, float h, int maxKey);
int findFirstKey(const device KeyIndex* keyIndices, int key, int low, int high);
int findLastKey(const device KeyIndex* keyIndices, int key, int low, int high);
#endif

#endif
