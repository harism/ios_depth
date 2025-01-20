#include <metal_stdlib>
#include "Types.h"
using namespace metal;

kernel void density(const device Vars& vars [[ buffer(0) ]],
                    const device int2* neighbours [[ buffer(1) ]],
                    device Particle* particles [[ buffer(2) ]],
                    uint gid [[ thread_position_in_grid ]])
{
    float density = vars.p0;
    Particle p = particles[gid];

    const float vars_m_poly6 = vars.m * vars.poly6;
    const float vars_h = vars.h;
    const float vars_h_2 = pow(vars_h, 2.0);
    
    for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
            int2 ii = neighbours[gid * 9 + y * 3 + z];
            for ( ; ii[0] >= 0 && ii[0] <= ii[1]; ++ii[0]) {
                const float r = distance(p.pos, particles[ii[0]].pos);
                if (r < vars_h) {
                    // density += vars.m * vars.poly6 * pow(pow(vars.h, 2.0) - pow(r, 2.0), 3.0);
                    density += vars_m_poly6 * pow(vars_h_2 - pow(r, 2.0), 3.0);
                }
            }
        }
    }
    
    p.density = density;
    p.pressure = max(0.0, vars.gasConstant * (density - vars.p0));
    particles[gid] = p;
}

kernel void forces(const device Vars& vars [[ buffer(0) ]],
                   const device int2* neighbours [[ buffer(1) ]],
                   device Particle* particles [[ buffer(2) ]],
                   uint gid [[ thread_position_in_grid ]])
{
    Particle p = particles[gid];
    float3 fPressure = float3(0.0);
    float3 fViscosity = float3(0.0);
    float3 fSurfaceNormal = float3(0.0);
    float fSurfaceLaplacian = 0.0;
    
    const float vars_spiky = vars.spikyLap;
    const float vars_h = vars.h;
    const float vars_visc = vars.viscGrad;
    const float vars_m = vars.m;
    const float vars_h_2 = pow(vars_h, 2.0);
    const float vars_poly6Grad = vars.poly6Grad;
    
    for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
            int2 ii = neighbours[gid * 9 + y * 3 + z];
            for ( ; ii[0] >= 0 && ii[0] <= ii[1]; ++ii[0]) {
                const Particle p2 = particles[ii[0]];
                const float3 v = p.pos - p2.pos;
                const float r = length(v);
                
                const float p2_density = p2.density;
                const float p2_one_per_density = 1.0 / p2_density;
                const float p2_m_per_density = vars_m * p2_one_per_density;
                
                if (r > 0.0 && r < vars_h) {
                    // const float3 wPressure = vars.spiky * pow(vars.h - r, 3.0) * normalize(v);
                    // fPressure -= vars.m * ((p.pressure + p2.pressure) / (2.0 * p2.density)) * wPressure;
                    //const float3 wPressure = (vars_spiky * r) * pow(vars_h - r, 2.0) * -v;
                    //fPressure -= vars_m * ((p.pressure + p2.pressure) / (2.0 * p2_density)) * wPressure;
                    
                    const float3 wPressure = vars_spiky * pow(vars_h - r, 2.0) * normalize(v);
                    fPressure -= vars_m * vars_m * ((p.pressure / pow(p.density, 2.0)) + (p2.pressure / pow(p2.density, 2.0))) * wPressure;
                    
                    // const float wVisc = vars.visc * (vars.h - r);
                    // fViscosity += vars.m * ((p2.vel - p.vel) / p2.density) * wVisc;
                    const float wVisc = vars_visc * (vars_h - r);
                    fViscosity += vars_m * ((p2.vel - p.vel) * p2_one_per_density) * wVisc;
                    
                    // const float3 wSurfGradient = vars.poly6Grad * pow(pow(vars.h, 2.0) - pow(r, 2.0), 2.0) * v;
                    // const float wSurfLaplacian = vars.poly6Grad * (pow(vars.h, 2.0) - pow(r, 2.0)) * (3.0 * pow(vars.h, 2.0) - 7.0 * pow(r, 2.0));
                    // fSurfaceNormal += (vars.m / p2.density) * wSurfGradient;
                    // fSurfaceLaplacian += (vars.m / p2.density) * wSurfLaplacian;
                    fSurfaceNormal += p2_m_per_density * vars.poly6Lap * -6.0 * pow(vars_h_2 - pow(r, 2.0), 2.0) * v;
                    fSurfaceLaplacian += p2_m_per_density * vars_poly6Grad * -6.0 * (3.0 * pow(vars_h, 4.0) - 10.0 * vars_h_2 * pow(r, 2.0) + 7.0 * pow(r, 4.0));      // (vars_h_2 - pow(r, 2.0)) * (3.0 * vars_h_2 - 7.0 * pow(r, 2.0));
                }
            }
        }
    }
    
    p.force = vars.viscosityConstant * fViscosity + fPressure;
    if (length(fSurfaceNormal) > 0.0) { // 7.065) {
        p.force += (-0.00728 * normalize(fSurfaceNormal) * (fSurfaceLaplacian));
    }
    p.force += p.density * float3(0.0, 0.0, 0.0);
    particles[gid] = p;
}
