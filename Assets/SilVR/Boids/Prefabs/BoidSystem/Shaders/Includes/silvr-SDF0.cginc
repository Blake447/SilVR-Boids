#include "silvr-raymarching.cginc"

float DE(float3 p)
{
    float3 z3 = float3(0, 0, 0);

    float sphere = -fSphere(p, 25, float3(0, 0, 0));
	//float cyl0 = fCapsule(p.zxy, 8.14, 4, float3(0, -4.2, -25.9));
    float cyl0 = fCapsule(p.zxy, 13.9, 5.77, float3(0.45, -6.58, -21.6));
    float box = fBox(p.zxy, float3(5.77, 9.04, 2.6), float3(0.45, -6.58 + 13.9 * 0.5, -21.6 + -2.92));
    float d = sphere;
    d = min(d, cyl0);
    d = smin(d, box, 5);
    
	//float box0 = fBox(p, float3(2.28, 2.597855, 1.827455)*1.2, float3(-0.46, -2.31, 3.01));
	//float sphere0 = fSphere(p, 1.83, float3(-4.79, -3.9 - 0.32, 3.08));
	//float cyl0 = fCapsule(p, 1.89, 0.51, float3(3.43, -2.58, 0.31));
#ifdef VISUALIZER
    return abs(d);
#else
    return d;
#endif
}

float3 calcNormal(float3 p, inout float f)
{
    float2 e = float2(1.0, -1.0) * .0005;
    return normalize(
	e.xyy * DE(p + e.xyy) +
	e.yyx * DE(p + e.yyx) +
	e.yxy * DE(p + e.yxy) +
	e.xxx * DE(p + e.xxx));
}