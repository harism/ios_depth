#include <metal_stdlib>
#include "Types.h"
using namespace metal;

kernel void euler(const device Vars& vars [[ buffer(0)]],
                  device Particle* particles [[ buffer(1) ]],
                  const device float& timeStep [[ buffer(2) ]],
                  //const device float4& gravity [[ buffer(3) ]],
                  //const device float& rms [[ buffer(4) ]],
                  const texture2d<float, access::sample> tx [[ texture(0) ]],
                  const texture2d<float, access::sample> depth [[ texture(1) ]],
                  uint gid [[ thread_position_in_grid ]])
{
    Particle p = particles[gid];
    
    float3 pos = p.pos;
    float3 vel = p.vel;

    float3 rot = 0.0;
    //rot.x = abs(pos.y) > 0.6 ? 0.0 : sign(pos.x) * (0.9 - abs(pos.x * 0.5)) * 40.0;
    //rot.y = abs(pos.x) > 1.6 ? 0.0 : sign(pos.y) * (0.9 - abs(pos.y)) * 40.0;
    //rot.z = -pos.z * 4.0;

    float3 a = (p.force + rot) / p.density;
    
    for (int i = 2; i < 4; ++i) {
        vel += a * timeStep;
        float3 dvel = vel * timeStep;

        if (abs(pos.x + dvel.x) > 1.0) {
            vel.x *= sign(vel.x) * sign(pos.x) * -0.1;
            dvel.x *= sign(dvel.x) * sign(pos.x) * -1.0;
            a.x *= sign(a.x) * sign(pos.x) * -1.0;
            pos.x = sign(pos.x) * 1.0;
        }
        
        if (abs(pos.y + dvel.y) > 2.0) {
            vel.y *= sign(vel.y) * sign(pos.y) * -0.1;
            dvel.y *= sign(dvel.y) * sign(pos.y) * -1.0;
            a.y *= sign(a.y) * sign(pos.y) * -1.0;
            pos.y = sign(pos.y) * 2.0;
        }
        
        if (abs(pos.z + dvel.z) > 0.5) {
            vel.z *= sign(vel.z) * sign(pos.z) * -0.5;
            dvel.z *= sign(dvel.z) * sign(pos.z) * -1.0;
            a.z *= sign(a.z) * sign(pos.z) * -1.0;
            pos.z = sign(pos.z) * 0.5;
        }
        
        pos += dvel;
        p.pos = pos;
    }
    
    float2 texPos = float2((p.oidx & 131071) / 256, p.oidx % 256) / float2(512.0, 256.0);
    //texPos.y = texPos.y * 0.8 + 0.1;
    constexpr sampler txSampler(mag_filter::bicubic, min_filter::bicubic);
    constexpr sampler depthSampler(mag_filter::bicubic, min_filter::bicubic);
    
    /*
    const float third = 1.0 / 3.0;
    const float third2 = 2.0 / 3.0;
    const float txX = texPos.x * third;
    const float txY = third * texPos.y;
    const float r = tx.sample(txSampler, float2(txX, txY)).r;
    const float g = tx.sample(txSampler, float2(txX, txY + third)).r;
    const float b = tx.sample(txSampler, float2(txX, txY + third2)).r;
    const float3 tex = saturate(float3(r, g, b) * 0.5 + 0.5);
     */
    const float3 tex = tx.sample(txSampler, texPos).rgb;
    const float d = depth.sample(depthSampler, texPos).r;
    const float3 pressure = saturate(float3(p.pressure / 100.0 * 10, p.pressure / 10.0 * 10.0, p.pressure / 10.0));
    
    if (p.oidx < 131072) {  // && rms * d < 100.0) {
        p.pos -= 0.3 * (p.pos - float3(float2(-1.0, -2.0) * (texPos * 2.0 - 1.0), -0.5 + d * 0.01));
        p.color = saturate(float4(tex, 1.0));
    } else {
        //p.pos -= 0.1 * (p.pos - float3(float2(-1.0, -2.0) * (texPos * 2.0 - 1.0), -0.2 + d * 1.0));
        p.color = saturate(float4(p.pressure * 0.0001, p.pressure * 0.00001, p.pressure * 0.000001, 0.01));
    }

    p.vel = vel;
    particles[gid] = p;
}
