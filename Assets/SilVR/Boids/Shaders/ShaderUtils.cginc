// Two Color conversion methods courtesy of 1001 from the VRC Shader community
float3 hsv2rgb(float x, float y, float z) {
    return z + z * y * (clamp(abs(fmod(x * 6. + float3(0, 4, 2), 6.) - 3.) - 1., 0., 1.) - 1.);
}
float3 rgb2hsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Two memory manipulation methods courtesy of Toocanzs, of the VRC Shader community as well
float4 float32ToRGBA8(float x)
{
    uint4 f = asuint(x);
    uint4 bytes = (f / uint4(0x1000000, 0x10000, 0x100, 1)) & uint4(0xff, 0xff, 0xff, 0xff);
    float4 output = float4(bytes);
    return output / 255;
}

float RGBA8ToFloat32(float4 f)
{
    f *= 255;
    uint4 bytes = (uint4(f) * uint4(0x1000000, 0x10000, 0x100, 1));
    uint full = bytes.x + bytes.y + bytes.z + bytes.w;
    return asfloat(full);
}