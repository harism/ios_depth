#include <metal_stdlib>
#include "Types.h"
using namespace metal;

template<int N> struct unrollReadLoop {
    static void read(const device KeyIndex* keyIndices, thread int* localKeys, int index) {
        localKeys[N - 1] = keyIndices[index + N - 1].key;
        unrollReadLoop<N - 1>::read(keyIndices, localKeys, index);
    }
    static void calc(const thread int* localKeys, int radix, thread uint* local) {
        ++local[(localKeys[N - 1] >> radix) & 0x0F];
        unrollReadLoop<N - 1>::calc(localKeys, radix, local);
    }
    static void call(const device KeyIndex* keyIndices, int index, int endIndex, int radix, thread uint* local) {
        int localKeys[N];
        while (index <= endIndex - N) {
            unrollReadLoop<N>::read(keyIndices, localKeys, index);
            unrollReadLoop<N>::calc(localKeys, radix, local);
            index += N;
        }
        unrollReadLoop<N / 2>::call(keyIndices, index, endIndex, radix, local);
    }
};
template <> struct unrollReadLoop<0> {
    static void read(const device KeyIndex* keyIndices, thread int* localKeys, int index) {}
    static void calc(const thread int* localKeys, int radix, thread uint* local) {}
    static void call(const device KeyIndex* keyIndices, int index, int endIndex, int radix, thread uint* local) {}
};

template<int N, int M = N> struct unrollWriteLoop {
    static void read(const device KeyIndex* keyIndices, thread KeyIndex* localKeys, int index) {
        localKeys[N - M] = keyIndices[index + N - M];
        unrollWriteLoop<N, M - 1>::read(keyIndices, localKeys, index);
    }
    static void calc(const thread KeyIndex* localKeys, thread int* localIndices, int index, int radix, thread uint* local) {
        const int r = (localKeys[N - M].key >> radix) & 0x0F;
        localIndices[N - M] = local[r]++;
        unrollWriteLoop<N, M - 1>::calc(localKeys, localIndices, index, radix, local);
    }
    static void save(device KeyIndex* keyIndicesTmp, thread KeyIndex* localKeys, thread int* localIndices) {
        keyIndicesTmp[localIndices[N - M]] = localKeys[N - M];
        unrollWriteLoop<N, M - 1>::save(keyIndicesTmp, localKeys, localIndices);
    }
    static void call(const device KeyIndex* keyIndices, device KeyIndex* keyIndicesTmp, int index, int endIndex, int radix, thread uint* local) {
        int localIndices[N];
        KeyIndex localKeys[N];
        while (index <= endIndex - N) {
            unrollWriteLoop<N>::read(keyIndices, localKeys, index);
            unrollWriteLoop<N>::calc(localKeys, localIndices, index, radix, local);
            unrollWriteLoop<N>::save(keyIndicesTmp, localKeys, localIndices);
            index += N;
        }
        unrollWriteLoop<N / 2>::call(keyIndices, keyIndicesTmp, index, endIndex, radix, local);
    }
};
template <int N> struct unrollWriteLoop<N, 0> {
    static void read(const device KeyIndex* keyIndices, thread KeyIndex* localKeys, int index) {}
    static void calc(const thread KeyIndex* localKeys, thread int* localIndices, int index, int radix, thread uint* local) {}
    static void save(device KeyIndex* keyIndicesTmp, thread KeyIndex* localKeys, thread int* localIndices) {}
    static void call(const device KeyIndex* keyIndices, device KeyIndex* keyIndicesTmp, int index, int endIndex, int radix, thread uint* local) {}
};

kernel void sortInit(const device Vars& vars [[ buffer(0) ]],
                     device Particle* particles [[ buffer(1) ]],
                     device KeyIndex* keyIndices [[ buffer(2) ]],
                     uint gid [[ thread_position_in_grid ]])
{
    const float3 pos = particles[gid].pos;
    KeyIndex keyIndex;
    keyIndex.key = getKey(getKey(pos, vars.simWidth, vars.h, vars.maxKey));
    keyIndex.index = gid;
    keyIndices[gid] = keyIndex;
}

kernel void sortRadix(const device Vars& vars [[ buffer(0) ]],
                      device KeyIndex* keyIndices [[ buffer(1) ]],
                      device KeyIndex* keyIndicesTmp [[ buffer(2) ]],
                      const device int& count [[ buffer(3) ]],
                      threadgroup uint* shared [[ threadgroup(0) ]],
                      uint gid [[ thread_position_in_threadgroup ]],
                      uint gsz [[ threads_per_threadgroup ]])
{
    const int size = (count + gsz - 1) / gsz;
    const int start = gid * size;
    const int end = min((gid + 1) * size, uint(count));
    
    for (int radix = 0; radix < 32; radix += 4) {
        
        uint local[16] = { 0 };
        
        {
            unrollReadLoop<32>::call(keyIndices, start, end, radix, local);
        }

        for (int k = 0; k < 4; ++k) {

            if ((gid & 0x03) == k) {
                for (int i = 0, o = gid >> 2; i < 16; ++i, o += gsz >> 2) {
                    if (k == 0) {
                        shared[o] = local[i];
                        local[i] = 0;
                    } else {
                        const int tmp = shared[o];
                        shared[o] = tmp + local[i];
                        local[i] = tmp;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        for (uint k = 1; k < gsz << 2; k <<= 1) {
            
            uint sum[4];

            for (uint i = 0, o = gid + k; i < 4 && o < (gsz << 2); ++i, o += gsz) {
                sum[i] = shared[o - k] + shared[o];
            }
            
            threadgroup_barrier(mem_flags::mem_none);
            
            for (uint i = 0, o = gid + k; i < 4 && o < (gsz << 2); ++i, o += gsz) {
                shared[o] = sum[i];
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        for (uint i = 0, o = gid >> 2; i < 16; ++i, o += gsz >> 2) {
            if (o > 0) {
                local[i] += shared[o - 1];
            }
        }
        
        {
            unrollWriteLoop<16>::call(keyIndices, keyIndicesTmp, start, end, radix, local);
            threadgroup_barrier(mem_flags::mem_device);
            
            device KeyIndex* tmp = keyIndices;
            keyIndices = keyIndicesTmp;
            keyIndicesTmp = tmp;
        }
    }
}

kernel void sortReorder(const device Vars& vars [[ buffer(0) ]],
                    device KeyIndex* keyIndices [[ buffer(1) ]],
                    const device Particle* particlesIn [[ buffer(2) ]],
                    device Particle* particlesOut [[ buffer(3) ]],
                    uint gid [[ thread_position_in_grid ]])
{
    KeyIndex keyIndex = keyIndices[gid];
    const Particle p = particlesIn[keyIndex.index];
    particlesOut[gid] = p;
}

kernel void sortNeighbours(const device Vars& vars [[ buffer(0) ]],
                           const device KeyIndex* keyIndices [[ buffer(1) ]],
                           device Particle* particles [[ buffer(2) ]],
                           device int2* neighbours [[ buffer(3) ]],
                           const device int& count [[ buffer(4) ]],
                           uint2 gid [[ thread_position_in_threadgroup ]],
                           uint2 gsz [[ threads_per_threadgroup ]],
                           uint2 ppos [[ threadgroup_position_in_grid ]])
{
    const int size = (count + gsz.x - 1) / gsz.x;
    int index = gid.x * size;
    const int endIndex = min((gid.x + 1) * size, uint(count));
    
    const float vars_simWidth = vars.simWidth;
    const float vars_h = vars.h;
    const int vars_maxKey = vars.maxKey;

    while (index < endIndex) {
        const int3 key = getKey(particles[index].pos, vars_simWidth, vars_h, vars_maxKey);
        int2 indices = int2(-1);
        int2 indicesPos = int2(key.y + ppos.x - 1, key.z + ppos.y - 1);
        const int keyInt = getKey(int3(key.x, indicesPos));
        
        if (all(indicesPos >= 0) && all(indicesPos <= vars_maxKey)) {
            int len = popcount(count) <= 1 ? count >> 1 : 1 << (31 - clz(count));
            int4 keys = int4(max(0, 1 - key.x) - 1, 0, int2(1, 2) - max(0, key.x + 1 - vars_maxKey)) + keyInt;
            int4 pos = select(int4(0), int4(count - len), int4(keyIndices[len].key) < keys);
            for ( ; len >= 1; len >>= 1) {
                pos[0] += select(0, len, keyIndices[pos[0] + len - 1].key < keys[0]);
                pos[1] += select(0, len, keyIndices[pos[1] + len - 1].key < keys[1]);
                pos[2] += select(0, len, keyIndices[pos[2] + len - 1].key < keys[2]);
                pos[3] += select(0, len, keyIndices[pos[3] + len - 1].key < keys[3]);
            }
            for (int i = 0; i < 3; ++i) {
                if (keyIndices[pos[i]].key == keys[i]) {
                    indices[0] = pos[i];
                    for (int j = 3; j > 0; --j) {
                        if (keyIndices[pos[j] - 1].key == keys[j - 1]) {
                            indices[1] = pos[j] - 1;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        
        int3 key2;
        neighbours[(index++) * 9 + ppos[0] * 3 + ppos[1]] = indices;
        while (index < endIndex && all(key == (key2 = getKey(particles[index].pos, vars_simWidth, vars_h, vars_maxKey)))) {
            neighbours[(index++) * 9 + ppos[0] * 3 + ppos[1]] = indices;
        }
    }
}

kernel void sortReorderOidx(const device Vars& vars [[ buffer(0) ]],
                            const device Particle* particlesIn [[ buffer(1) ]],
                            device Particle* particlesOut [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]])
{
    const Particle p = particlesIn[gid];
    particlesOut[p.oidx] = p;
}
