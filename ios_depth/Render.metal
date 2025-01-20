#include <metal_stdlib>
#include "Types.h"
using namespace metal;

struct CopyData
{
    float4 position [[position]];
    float2 texPos;
};

vertex CopyData copyVertexShader(uint vertexId [[ vertex_id ]])
{
    const float2 pos = float2((vertexId >> 1), (vertexId & 1));

    CopyData out;
    out.position = float4(float2(0.0 + pos.x, 1.0 - pos.y) * 2.0 - 1.0, 0.0, 1.0);
    out.texPos = pos;
    return out;
}

fragment float4 copyFragmentShader(CopyData copyData [[ stage_in ]],
                                   texture2d<float> tx [[ texture(0) ]])
{
    constexpr sampler textureSampler(mag_filter::bicubic, min_filter::bicubic);
    const float4 txCol = tx.sample(textureSampler, copyData.texPos);
    return txCol;
}

struct VertexData
{
    float4 position [[position]];
    float2 positionRel;
    float4 color;
};

vertex VertexData renderParticleVertexShader(constant Vars& vars [[ buffer(0) ]],
                                             constant Particle* particles [[ buffer(1) ]],
                                             //constant int& count [[ buffer(2) ]],
                                             //constant float4x4& projMatrix [[ buffer(3) ]],
                                             //constant float4x4& viewModelMatrix [[ buffer(4) ]],
                                             //texture2d<float, access::sample> tx [[ texture(0) ]],
                                             uint vertexId [[ vertex_id]],
                                             uint instanceId [[ instance_id ]])
{
    constexpr sampler txSampler(mag_filter::bicubic, min_filter::bicubic);

    VertexData out;
    //const Particle p = particles[instanceId + vertexId * 262144]; // + (vertexId >> 1) + (vertexId & 1) * 262144];
    const Particle p = particles[instanceId];
    const float3 ppos = p.pos;

    /*
    float alpha = 0.85;
    
    const float3 ppos2 = particles[instanceId + (vertexId + 1) % 3].pos[rit % ritCount];
    const float3 ppos3 = particles[instanceId + (vertexId + 2) % 3].pos[rit % ritCount];
    
    const float distp2 = distance(ppos, ppos2);
    if (distp2 > 0.1) {
        alpha = mix(alpha, 0.0, saturate((distp2 - 0.1) * 8.0));
    }
    
    const float distp3 = distance(ppos, ppos3);
    if (distp3 > 0.1) {
        alpha = mix(alpha, 0.0, saturate((distp3 - 0.1) * 8.0));
    }

    const float distp23 = distance(ppos2, ppos3);
    if (distp23 > 0.1) {
        alpha = mix(alpha, 0.0, saturate((distp23 - 0.1) * 8.0));
    }
    
    if (alpha <= 0.0) {
        out.position.xyz = float3(-1000000);
        out.position.w = 1.0;
        return out;
    }
     */

    /*
    const float third = 1.0 / 3.0;
    const float third2 = 2.0 / 3.0;
    const float txX = p.tex.x * third;
    const float txY = third * (1.0 - p.tex.y);
    const float r = tx.sample(txSampler, float2(txX, txY)).r;
    const float g = tx.sample(txSampler, float2(txX, txY + third)).r;
    const float b = tx.sample(txSampler, float2(txX, txY + third2)).r;
    float3 tex = float3(r, g, b) + 0.3;
     */
    //float3 tex = float3(0.020, 0.022, 0.028) * 10.0;
    //float3 tex = float3(0.02, 0.022, 0.021);
    
    const float2 posIdx = float2(vertexId >> 1, vertexId & 1);
    const float2 posRel = float2(0.0 + posIdx.x, 1.0 - posIdx.y) * 2.0 - 1.0;
    //float4 posView = viewModelMatrix * float4(ppos, 1.0);

    //posView.xy += posRel * (vars.d + p.oidx >= 262144 ? 0.01 : 0.0);
    out.position = float4(ppos.x + posRel.x * vars.d, ppos.y * 0.5 + posRel.y * vars.d, 0.0, 1.0);   //projMatrix * posView;
    out.positionRel = posRel;
    
    //float4 tex = tx.sample(txSampler, ((float2(1.0, -1.0) * out.position.xy) / out.position.w) * 0.5 + 0.5);
    //out.color = saturate( float4(mix(tex.rgb * 1.0, tex.rgb * 1.75, ((1.0 - ppos.z) * 0.5) < tex.a ? 0.0 : 1.0), tex.a) );
    
    out.color = p.color; // alpha);
    
    return out;
}

fragment float4 renderParticleFragmentShader(VertexData vertexData [[ stage_in ]])
{
    //const float dist = length(vertexData.positionRel);
    //const float4 color = float4(vertexData.color.rgb, vertexData.color.a * saturate(1.0 - dist * dist));
    //return float4(color.rgb, (1.0 - dist) * color.a);
    return float4(vertexData.color.rgb, vertexData.color.a);
}
