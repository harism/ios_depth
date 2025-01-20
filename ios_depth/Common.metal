#include <metal_stdlib>
#include "Types.h"
using namespace metal;

int getKey(int3 key) {
    return key.x & 0x3FF | (key.y & 0x3FF) << 10 | (key.z & 0x3FF) << 20;
}

int3 getKey(float3 pos, float simWidth, float h, int maxKey) {
    const float3 p = saturate((pos + (simWidth * 0.5) - h) * 0.25);
    return int3(p * maxKey);
}

int findFirstKey(const device KeyIndex* keyIndices, int key, int low, int high)
{
    int len = high - low + 1;
    len = popcount(len) <= 1 ? len >> 1 : 1 << (31 - clz(len));
    int pos = keyIndices[low + len].key < key ? high - len + 1 : low;
    for ( ; len >= 8; len >>= 4) {
        pos += select(0, len, keyIndices[pos + len - 1].key < key);
        pos += select(0, len >> 1, keyIndices[pos + (len >> 1) - 1].key < key);
        pos += select(0, len >> 2, keyIndices[pos + (len >> 2) - 1].key < key);
        pos += select(0, len >> 3, keyIndices[pos + (len >> 3) - 1].key < key);
    }
    for ( ; len >= 4; len >>= 3) {
        pos += select(0, len, keyIndices[pos + len - 1].key < key);
        pos += select(0, len >> 1, keyIndices[pos + (len >> 1) - 1].key < key);
        pos += select(0, len >> 2, keyIndices[pos + (len >> 2) - 1].key < key);
    }
    for ( ; len >= 2; len >>= 2) {
        pos += select(0, len, keyIndices[pos + len - 1].key < key);
        pos += select(0, len >> 1, keyIndices[pos + (len >> 1) - 1].key < key);
    }
    for ( ; len >= 1; len >>= 1) {
        pos += select(0, len, keyIndices[pos + len - 1].key < key);
    }
    return select(-1, pos, keyIndices[pos].key == key);
}

int findLastKey(const device KeyIndex* keyIndices, int key, int low, int high)
{
    int len = high - low + 1;
    len = popcount(len) <= 1 ? len >> 1 : 1 << (31 - clz(len));
    int pos = keyIndices[low + len].key < key ? high : low + len - 1;
    for ( ; len >= 8; len >>= 4) {
        pos -= select(0, len, keyIndices[pos - len + 1].key > key);
        pos -= select(0, len >> 1, keyIndices[pos - (len >> 1) + 1].key > key);
        pos -= select(0, len >> 2, keyIndices[pos - (len >> 2) + 1].key > key);
        pos -= select(0, len >> 3, keyIndices[pos - (len >> 3) + 1].key > key);
    }
    for ( ; len >= 4; len >>= 3) {
        pos -= select(0, len, keyIndices[pos - len + 1].key > key);
        pos -= select(0, len >> 1, keyIndices[pos - (len >> 1) + 1].key > key);
        pos -= select(0, len >> 2, keyIndices[pos - (len >> 2) + 1].key > key);
    }
    for ( ; len >= 2; len >>= 2) {
        pos -= select(0, len, keyIndices[pos - len + 1].key > key);
        pos -= select(0, len >> 1, keyIndices[pos - (len >> 1) + 1].key > key);
    }
    for ( ; len >= 1; len >>= 1) {
        pos -= select(0, len, keyIndices[pos - len + 1].key > key);
    }
    return select(-1, pos, keyIndices[pos].key == key);
}
